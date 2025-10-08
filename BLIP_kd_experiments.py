import yaml
import wandb
import torch
import argparse

import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode

from simple_tokenizer import tokenize, SimpleTokenizer
from clip_model import build_CLIP_from_openai_pretrained

from data.flickr30k_dataset import ps_train_dataset_KD, ps_eval_dataset_KD
from models.blip_retrieval import BLIP_Retrieval

"""
1. custom dataset 만들기
   - RaSa의 dual encoder로 먼저 image, text embedding 다 추출해서 저장해놓기(이게 내 데이터) ==> (O)
   - student model용 dataset 만들기(IRRA 참고) ==> 이때 데이터 Path를 꼭 살려놔야 함. ==> (O)
2. CLIP으로 pretrained된 모델 불러오기 ==> (O)
3. RaSa 모델의 ITM하기 위한 encoder만 불러와서 Teacher model로 사용 ==> (O)
4. RaSa 모델의 ITM score 분포를 Student model이 KL or MSE로 학습할 수 있도록 training loop 작성
5. 학습 완료 후에는 모델 evaluation on Test set ==> 이때는 rasa의 evaluation 코드 참고하기.(dual encoder 부분만 참고하면 될듯)
"""

@torch.no_grad()
def get_teacher_distribution(teacher_model_itm_head, teacher_model_cross_encoder, text_ids, text_attentions, image_feats, image_attentions, text_embeds, image_embeds,device):
   
   text_ids = text_ids.to(device)
   text_attentions = text_attentions.to(device)
   text_embeds = text_embeds.to(device)
   
   image_feats = image_feats.to(device)
   image_attentions = image_attentions.to(device)   
   image_embeds = image_embeds.to(device)
   
   sim_matrix = text_embeds @ image_embeds.t()  # (num_texts, num_images)
   
   # 주의! cross encoder자체가 cross attention을 하는 모델이라, 엄밀히 말하면 t2i를 transpose한다고 해서 i2t가 아닐 수 있음.
   num_texts = text_ids.shape[0]
   num_images = image_feats.shape[0]
   score_matrix_t2i = torch.zeros(num_texts, num_images, device=device, requires_grad=False)

   for i in range(num_texts):
      cur_text_ids = text_ids[i].repeat(num_images, 1)  # (num_images, seq_len, hidden)
      cur_text_atts  = text_attentions[i].repeat(num_images, 1)  # (num_images, seq_len)

      output = teacher_model_cross_encoder(
         cur_text_ids,
         attention_mask=cur_text_atts,
         encoder_hidden_states=image_feats,  # (num_images, img_seq_len, hidden)
         encoder_attention_mask=image_attentions,
         return_dict=True,
      )
      logits = teacher_model_itm_head(output.last_hidden_state[:, 0, :])  # (num_images, 2)
      itm_scores = logits[:, 1]  # Positive class logit (not probability)
      #Warning : 원래 코드에서는 score + dual_encoder_score를 더했는데, dual encoder score는 없으니 그냥 score만 사용함. 
      dual_scores = sim_matrix[i]  # (num_images,) ← text i 기준 similarity vector

      # # 최종 score = ITM logit + dual encoder similarity (선형 결합, 필요 시 가중치 조정 가능)
      final_scores = itm_scores + dual_scores

      score_matrix_t2i[i] = itm_scores
      
   return score_matrix_t2i
  
def get_student_distribution(student_model_clip, images, caption_ids, logit_scale, device, return_feats=False):
   
   images = images.to(device)
   caption_ids = caption_ids.to(device)
   image_feats, text_feats = student_model_clip(images, caption_ids)
   i_feats = image_feats[:, 0, :].float()
   t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
   #t2i_logits = logit_scale * (t_feats @ i_feats.t()) ## for MSE case
   #TODO: 이것도 성능이 안 나오면 t_feats와 i_feats를 normalize해서 cosine similarity로 바꿔서 해보기.
   t2i_logits = torch.exp(logit_scale) * (t_feats @ i_feats.t())  # for KL_divergence case

   if return_feats:
      return t2i_logits, t_feats, i_feats
   
   return t2i_logits

def get_contrastive_loss(t_feats, i_feats, logit_scale):
   
   t_feats = F.normalize(t_feats, dim=-1)
   i_feats = F.normalize(i_feats, dim=-1)
   logits_per_text = torch.exp(logit_scale) * t_feats @ i_feats.t()
   logits_per_image = logits_per_text.t()
   labels = torch.arange(t_feats.size(0), device=t_feats.device)
   loss_t2i = F.cross_entropy(logits_per_text,  labels)
   loss_i2t = F.cross_entropy(logits_per_image, labels)
   
   return (loss_t2i + loss_i2t) / 2

def get_sdm_loss(t_feats, i_feats, pids, logit_scale):
   """
   Similarity Distribution Matching Loss from IRRA paper
   """   
   batch_size = t_feats.shape[0]
   pids = pids.reshape(batch_size, 1)  # (batch_size, 1)
   pid_dist = pids - pids.t()
   labels = (pid_dist == 0).float().to(t_feats.device)  # (batch_size, batch_size)
   
   t_feats = F.normalize(t_feats, dim=-1)
   i_feats = F.normalize(i_feats, dim=-1)
   
   logits_per_text  = torch.exp(logit_scale) * t_feats @ i_feats.t()
   logits_per_image = logits_per_text.t()
   
   labels_distribute = labels / labels.sum(dim=1)
   epsilon = 1e-8
   loss_t2i = F.kl_div(F.log_softmax(logits_per_text, dim=-1),
                       labels_distribute + epsilon, 
                       reduction='batchmean')
   loss_i2t = F.kl_div(F.log_softmax(logits_per_image, dim=-1),
                       labels_distribute + epsilon, 
                       reduction='batchmean')
   loss = (loss_t2i + loss_i2t) / 2

   return loss

@torch.no_grad()
def itm_eval(scores_t2i, img2person, txt2person, eval_mAP):
   
   img2person  = torch.tensor(img2person)
   txt2person  = torch.tensor(txt2person)
   index       = torch.argsort(scores_t2i, dim=-1, descending=True)
   pred_person = img2person[index]
   matches     = (txt2person.view(-1, 1).eq(pred_person)).long()

   def acc_k(matches, k=1):
      matches_k = matches[:, :k].sum(dim=-1)
      matches_k = torch.sum((matches_k > 0))
      return 100.0 * matches_k / matches.size(0)

   # Compute metrics
   ir1     = acc_k(matches, k=1).item()
   ir5     = acc_k(matches, k=5).item()
   ir10    = acc_k(matches, k=10).item()
   ir_mean = (ir1 + ir5 + ir10) / 3

   if eval_mAP:
      real_num = matches.sum(dim=-1)
      tmp_cmc  = matches.cumsum(dim=-1).float()
      order    = torch.arange(start=1, end=matches.size(1) + 1, dtype=torch.long)
      tmp_cmc /= order
      tmp_cmc *= matches
      AP  = tmp_cmc.sum(dim=-1) / real_num
      mAP = AP.mean() * 100.0
      eval_result = {'r1': ir1, 
                     'r5': ir5,
                     'r10': ir10, 
                     'mAP': mAP.item()}
   else:
      eval_result = {'r1': ir1, 
                     'r5': ir5,
                     'r10': ir10, 
                     'r_mean': ir_mean}
   return eval_result
   
@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, logit_scale):
    # evaluate
    model.eval()
    print('Computing features for evaluation...')
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 64*2
    text_embeds = []
    caption_id_list = []
   
    for text in tqdm(texts, desc="Extracting text ids..."):
       caption_id_list.append(tokenize(text, tokenizer, text_length=77, truncate=True))
   
    for i in tqdm(range(0, len(caption_id_list), text_bs), desc="Extracting text features..."):
       caption_ids = caption_id_list[i: min(num_text, i + text_bs)]
       caption_ids = torch.stack(caption_ids, dim=0).to(device)
       text_feats  = model.encode_text(caption_ids)   
       t_feats     = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
       #t_norm  = t_feats / t_feats.norm(dim=1, keepdim=True)
       #text_embeds.append(t_norm)
       text_embeds.append(t_feats)
      
    text_embeds = torch.cat(text_embeds, dim=0)

    # extract image features
    image_embeds = []
    for images, img_id in tqdm(data_loader, desc="Extracting image features..."):
       images      = images.to(device)
       image_feats = model.encode_image(images)
       i_feats     = image_feats[:, 0, :].float()
       #i_norm = i_feats / i_feats.norm(dim=1, keepdim=True)
       #image_embeds.append(i_norm)
       image_embeds.append(i_feats)

    image_embeds = torch.cat(image_embeds, dim=0)

    # compute the feature similarity score for all image-text pairs
    sims_matrix = logit_scale * (text_embeds @ image_embeds.t())
    text_embeds_norm   = F.normalize(text_embeds,  dim=-1)
    image_embeds_norm  = F.normalize(image_embeds, dim=-1)
    sims_matrix_cosine = logit_scale * (text_embeds_norm @ image_embeds_norm.t())
    return sims_matrix.cpu(), sims_matrix_cosine.cpu()

def main(args, config):

    student_model_clip, clip_model_cfg = build_CLIP_from_openai_pretrained(name="ViT-B/16", image_size=(384, 128), stride_size=16, mask_ratio=0.0)
    student_model_clip.train()

    # TODO Learnable로 만들수도 있음
    #logit_scale_distill  = nn.Parameter(torch.ones([]) * torch.tensor(1.0))  # For MSE
    logit_scale_distill  = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))  # For KL_divergence
    logit_scale_contrast = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))  # For contrastive

    train_transform = transforms.Compose([
       transforms.Resize((384, 128), interpolation=InterpolationMode.BICUBIC),
       transforms.RandomHorizontalFlip(0.5),
       transforms.ToTensor(),
       transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                            std =(0.26862954, 0.26130258, 0.27577711)),
       transforms.RandomErasing(scale=(0.02, 0.4), value=[0.48145466, 0.4578275, 0.40821073]),
    ])
    test_transform = transforms.Compose([
       transforms.Resize((384, 128), interpolation=InterpolationMode.BICUBIC),
       transforms.ToTensor(),
       transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                           std =(0.26862954, 0.26130258, 0.27577711)),
    ])

    train_dataset = ps_train_dataset_KD(ann_file=args.train_annotation_file, transform=train_transform, embedding_path=args.embedding_path, image_root=args.image_root)
    test_dataset  = ps_eval_dataset_KD (ann_file=args.test_annotation_file,  transform=test_transform,  image_root=args.image_root)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size,     shuffle=True,  num_workers=args.num_workers)
    test_loader   = DataLoader(test_dataset,  batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers)
   
    optimizer = torch.optim.AdamW(list(student_model_clip.parameters()) + [logit_scale_distill, logit_scale_contrast], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    model = BLIP_Retrieval(image_size=config['image_res'], vit=config['vit'], 
                    vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                    queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']
    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % args.checkpoint_path)
    print(msg)
    
    teacher_model_cross_encoder = model.text_encoder
    teacher_model_itm_head      = model.itm_head
    enc_token_id                = model.tokenizer.enc_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model_cross_encoder.to(device)
    teacher_model_itm_head.to(device)
    student_model_clip.to(device)
   
    teacher_model_cross_encoder.eval()
    teacher_model_itm_head.eval()
   
    del model
    torch.cuda.empty_cache()
   
    wandb.init(project="BLIP_kd_experiment", 
               name=args.exp_name,
               notes =args.wandb_notes,
               config=args)
   
    alpha = 0.5  # distill vs contrastive 가중치
   
    for epoch in range(args.epochs):
       total_loss = 0.0
       student_model_clip.train()
       optimizer.zero_grad()
       for batch in tqdm(train_loader):
          optimizer.zero_grad()
          images, caption_ids, person_ids, text_ids, text_attentions, image_feats, image_attentions, text_embeds, image_embeds = batch
          text_ids[:,0] = enc_token_id
          teacher_logits = get_teacher_distribution(teacher_model_itm_head, teacher_model_cross_encoder, text_ids, text_attentions, image_feats, image_attentions, text_embeds, image_embeds, device)
          student_logits, t_feats, i_feats = get_student_distribution(student_model_clip, images, caption_ids, logit_scale_distill, device, return_feats=True)

          #distillation_loss = F.mse_loss(student_logits, teacher_logits)
          distillation_loss = F.kl_div(F.log_softmax(student_logits, dim=-1),
                                       F.softmax(teacher_logits, dim=-1), 
                                       reduction='batchmean')
          contrastive_loss = get_contrastive_loss(t_feats, i_feats, logit_scale_contrast)
          #contrastive_loss = get_sdm_loss(t_feats, i_feats, person_ids, logit_scale_contrast)         
          loss = alpha * distillation_loss + (1 - alpha) * contrastive_loss
          loss.backward()
          optimizer.step()         
          total_loss += loss.item()
         
          wandb.log({
            "step_distill_loss": distillation_loss.item(),
            "step_contrastive_loss": contrastive_loss.item()
            })
       print(f"[Epoch {epoch+1}] Average Loss: {total_loss / len(train_loader):.4f}")
       wandb.log({
          "epoch_loss": total_loss / len(train_loader),
          "epoch_distill_loss": distillation_loss.item(),
          "epoch_contrastive_loss": contrastive_loss.item(),
          "lr": scheduler.get_last_lr()[0]
          })

       scheduler.step()
       if (epoch + 1) % args.eval_every == 0:
          student_model_clip.eval()
          _, sim_matrix_cos = evaluate(student_model_clip, test_loader, SimpleTokenizer(), device, logit_scale_contrast)
          #eval_result_euc = itm_eval(sim_matrix_euc, test_dataset.img2person, test_dataset.txt2person, eval_mAP=True)
          eval_result_cos = itm_eval(sim_matrix_cos, test_dataset.img2person, test_dataset.txt2person, eval_mAP=True)
         
          wandb.log({
             'r1_cos': eval_result_cos['r1'],
             'r5_cos': eval_result_cos['r5'],
             'r10_cos': eval_result_cos['r10'],
             'mAP_cos': eval_result_cos['mAP']
             })
          print(f"[Epoch {epoch+1}] eval", eval_result_cos)

if __name__ == "__main__":
   
   parser = argparse.ArgumentParser()
   ## Annotation files
   parser.add_argument("--train_annotation_file",type=list, default=["/home/sooyoung/ps/CUHK-PEDES/processed_data/train.json"])
   parser.add_argument("--test_annotation_file", type=str,  default="/home/sooyoung/ps/CUHK-PEDES/processed_data/test.json")
   parser.add_argument("--config_path",          type=str,  default="/home/sooyoung/ps/BLIP/configs/retrieval_CUHK.yaml")
   
   ## Dataset & DataLoader
   parser.add_argument("--image_root",     type=str, default="/home/sooyoung/ps/CUHK-PEDES/imgs")
   parser.add_argument("--embedding_path", type=str, default="/home/sooyoung/ps/BLIP/BLIP_top1_75.58_pre-computed_embeddings")
   parser.add_argument("--batch_size",     type=int, default=64)
   parser.add_argument("--num_workers",    type=int, default=4)
   
   ## Teacher model
   parser.add_argument("--checkpoint_path", type=str, default="/home/sooyoung/ps/BLIP/checkpoint_best.pth")
   
   ## Training
   parser.add_argument("--epochs",    type=int,   default=60)
   parser.add_argument("--lr",        type=float, default=1e-5)
   parser.add_argument("--scheduler", type=str,   default="cosine")
   parser.add_argument("--eval_every",type=int,   default=1)
   
   ## Wandb
   parser.add_argument("--exp_name",    type=str, default=None, required=True)
   parser.add_argument("--wandb_notes", type=str, default=None, required=True)
   args = parser.parse_args()
   config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
   main(args, config)