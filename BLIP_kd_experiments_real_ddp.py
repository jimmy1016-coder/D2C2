import yaml
import wandb
import torch
import argparse

import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.functional import InterpolationMode
from torch.nn.parallel import DistributedDataParallel as DDP

from simple_tokenizer import tokenize, SimpleTokenizer
from clip_model import build_CLIP_from_openai_pretrained

from data.flickr30k_dataset import ps_train_dataset_KD, ps_eval_dataset_KD
from models.blip_retrieval import BLIP_Retrieval


class GatherLayer(torch.autograd.Function):
    """GatherLayer gathers input tensors from all processes, with backward support."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        if dist.is_initialized():
            output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
            dist.all_gather(output, input)
        else:
            output = [input]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        if dist.is_initialized():
            dist_ops = [
                dist.reduce(grads[i], i, async_op=True)
                for i in range(dist.get_world_size())
            ]
            for op in dist_ops:
                op.wait()
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()] if dist.is_initialized() else 0
        return grad_out

# ----------------------------
# Loss Wrapper (logit_scale + ID classifier)
# ----------------------------
class LossWrapper(nn.Module):
    def __init__(self, embed_dim, num_classes, use_id_loss=False):
        super().__init__()
        self.logit_scale_distill  = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        self.logit_scale_contrast = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        self.use_id_loss = use_id_loss
        if use_id_loss:
            self.id_classifier = nn.Linear(embed_dim, num_classes)
            nn.init.normal_(self.id_classifier.weight.data, std=0.001)
            nn.init.constant_(self.id_classifier.bias.data, val=0.0)

    def get_id_loss(self, t_feats, i_feats, person_ids):
        if not self.use_id_loss:
            return torch.tensor(0.0, device=t_feats.device)
        person_ids = person_ids.to(t_feats.device)
        image_logits = self.id_classifier(F.normalize(i_feats, dim=-1))
        text_logits  = self.id_classifier(F.normalize(t_feats, dim=-1))
        loss_fn = nn.CrossEntropyLoss()
        return (loss_fn(image_logits, person_ids) + loss_fn(text_logits, person_ids)) / 2


# ----------------------------
# Teacher distribution (BLIP cross-encoder)
# ----------------------------
@torch.no_grad()
def get_teacher_distribution(teacher_model_itm_head, teacher_model_cross_encoder,
                             text_ids, text_attentions, image_feats, image_attentions,
                             text_embeds, image_embeds, device):
    text_ids = text_ids.to(device)
    text_attentions = text_attentions.to(device)
    text_embeds = text_embeds.to(device)

    image_feats = image_feats.to(device)
    image_attentions = image_attentions.to(device)
    image_embeds = image_embeds.to(device)

    sim_matrix = text_embeds @ image_embeds.t()
    num_texts = text_ids.shape[0]
    num_images = image_feats.shape[0]
    score_matrix_t2i = torch.zeros(num_texts, num_images, device=device, requires_grad=False)

    for i in range(num_texts):
        cur_text_ids  = text_ids[i].repeat(num_images, 1)
        cur_text_atts = text_attentions[i].repeat(num_images, 1)
        output = teacher_model_cross_encoder(
            cur_text_ids,
            attention_mask=cur_text_atts,
            encoder_hidden_states=image_feats,
            encoder_attention_mask=image_attentions,
            return_dict=True,
        )
        logits = teacher_model_itm_head(output.last_hidden_state[:, 0, :])  # (num_images, 2)
        itm_scores = logits[:, 1]
        dual_scores = sim_matrix[i]
        
        score_matrix_t2i[i] = itm_scores + dual_scores
    return score_matrix_t2i


# ----------------------------
# Student distribution
# ----------------------------
def get_student_distribution(student_model_clip, images, caption_ids, logit_scale, device, return_feats=False):
    images = images.to(device)
    caption_ids = caption_ids.to(device)
    image_feats, text_feats = student_model_clip(images, caption_ids)
    i_feats = image_feats[:, 0, :].float()
    t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
    t2i_logits = torch.exp(logit_scale) * (t_feats @ i_feats.t())
    if return_feats:
        return t2i_logits, t_feats, i_feats
    return t2i_logits


# ----------------------------
# Contrastive loss
# ----------------------------
# def get_contrastive_loss(t_feats, i_feats, logit_scale):
#     t_feats = F.normalize(t_feats, dim=-1)
#     i_feats = F.normalize(i_feats, dim=-1)
#     logits_per_text = torch.exp(logit_scale) * t_feats @ i_feats.t()
#     logits_per_image = logits_per_text.t()
#     labels = torch.arange(t_feats.size(0), device=t_feats.device)
#     loss_t2i = F.cross_entropy(logits_per_text, labels)
#     loss_i2t = F.cross_entropy(logits_per_image, labels)
#     return (loss_t2i + loss_i2t) / 2

# Contrastive loss DDP version
def get_contrastive_loss(t_feats, i_feats, logit_scale):
    # normalize
    t_feats = F.normalize(t_feats, dim=-1)
    i_feats = F.normalize(i_feats, dim=-1)

    # gather across all GPUs (autograd-safe)
    t_feats_all = torch.cat(GatherLayer.apply(t_feats), dim=0)
    i_feats_all = torch.cat(GatherLayer.apply(i_feats), dim=0)

    # logits: local batch × global batch
    logits_per_text  = torch.exp(logit_scale) * (t_feats @ i_feats_all.t())
    logits_per_image = torch.exp(logit_scale) * (i_feats @ t_feats_all.t())

    # labels: offset by rank
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_bs = t_feats.size(0)
    labels = torch.arange(local_bs, device=t_feats.device) + rank * local_bs

    loss_t2i = F.cross_entropy(logits_per_text, labels)
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    return (loss_t2i + loss_i2t) / 2


# ----------------------------
# Evaluation functions
# ----------------------------
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

    ir1, ir5, ir10 = acc_k(matches,1).item(), acc_k(matches,5).item(), acc_k(matches,10).item()
    if eval_mAP:
        real_num = matches.sum(dim=-1)
        tmp_cmc  = matches.cumsum(dim=-1).float()
        order    = torch.arange(start=1, end=matches.size(1)+1, dtype=torch.long)
        tmp_cmc /= order
        tmp_cmc *= matches
        AP  = tmp_cmc.sum(dim=-1) / real_num
        mAP = AP.mean() * 100.0
        return {'r1':ir1, 'r5':ir5, 'r10':ir10, 'mAP':mAP.item()}
    return {'r1':ir1, 'r5':ir5, 'r10':ir10, 'r_mean':(ir1+ir5+ir10)/3}


@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, logit_scale):
    model.eval()
    print('Computing features for evaluation...')
    texts = data_loader.dataset.text
    text_bs = 128
    caption_id_list = [tokenize(t, tokenizer, text_length=77, truncate=True) for t in texts]

    text_embeds = []
    for i in tqdm(range(0, len(caption_id_list), text_bs), desc="Text features"):
        caption_ids = torch.stack(caption_id_list[i: i+text_bs], dim=0).to(device)
        text_feats  = model.encode_text(caption_ids)
        t_feats     = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        text_embeds.append(t_feats)
    text_embeds = torch.cat(text_embeds, dim=0)

    image_embeds = []
    for images, _ in tqdm(data_loader, desc="Image features"):
        images = images.to(device)
        image_feats = model.encode_image(images)
        i_feats = image_feats[:, 0, :].float()
        image_embeds.append(i_feats)
    image_embeds = torch.cat(image_embeds, dim=0)

    text_embeds_norm = F.normalize(text_embeds, dim=-1)
    image_embeds_norm = F.normalize(image_embeds, dim=-1)
    
    sims_matrix = logit_scale * (text_embeds_norm @ image_embeds_norm.t())
    return sims_matrix.cpu()


# ----------------------------
# Main training loop
# ----------------------------
def main(args, config):
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    device = torch.device(f'cuda:{local_rank}')

    # Data
    train_transform = transforms.Compose([
        transforms.Resize((384,128), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466,0.4578275,0.40821073),
                             std=(0.26862954,0.26130258,0.27577711)),
        transforms.RandomErasing(scale=(0.02,0.4), value=[0.48145466,0.4578275,0.40821073]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((384,128), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466,0.4578275,0.40821073),
                             std=(0.26862954,0.26130258,0.27577711)),
    ])
    train_dataset = ps_train_dataset_KD(ann_file=args.train_annotation_file, transform=train_transform, embedding_path=args.embedding_path, image_root=args.image_root)
    test_dataset  = ps_eval_dataset_KD (args.test_annotation_file, test_transform, args.image_root)

    train_sampler = DistributedSampler(train_dataset)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)
    test_loader   = DataLoader(test_dataset,  batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers)

    # Student
    student_model_clip, _ = build_CLIP_from_openai_pretrained(name="ViT-B/16", image_size=(384,128), stride_size=16, mask_ratio=0.0)
    loss_module = LossWrapper(embed_dim=512, num_classes=train_dataset.num_ids, use_id_loss=args.use_id_loss)  # num_classes는 dataset에 맞게 수정
    student_model = nn.ModuleDict({"student": student_model_clip, "losses": loss_module})

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    student_model.to(local_rank)
    student_model = DDP(student_model, device_ids=[local_rank], find_unused_parameters=True)

    # Teacher
    model = BLIP_Retrieval(image_size=config['image_res'], vit=config['vit'],
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                           queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    if dist.get_rank()==0:
        print("Loaded teacher:", msg)

    teacher_model_cross_encoder = model.text_encoder.to(device).eval()
    teacher_model_itm_head      = model.itm_head.to(device).eval()
    enc_token_id                = model.tokenizer.enc_token_id
    del model
    torch.cuda.empty_cache()

    if dist.get_rank()==0:
        wandb.init(project="BLIP_kd_experiment",
                   name=args.exp_name,
                   notes=args.wandb_notes,
                   config=args)
        
    # Training loop
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        total_loss=0.0
        student_model.train()
        if dist.get_rank()==0: pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        else: pbar = train_loader

        for batch in pbar:
            images, caption_ids, person_ids, text_ids, text_atts, image_feats, image_atts, text_embeds, image_embeds = batch
            caption_ids, text_ids = caption_ids.to(device), text_ids.to(device)
            text_ids[:,0] = enc_token_id

            # Forward teacher & student
            teacher_logits = get_teacher_distribution(teacher_model_itm_head, teacher_model_cross_encoder,
                                                      text_ids, text_atts, image_feats, image_atts,
                                                      text_embeds, image_embeds, device)
            student_logits, t_feats, i_feats = get_student_distribution(student_model.module.student,
                                                                        images, caption_ids,
                                                                        student_model.module.losses.logit_scale_distill,
                                                                        device, return_feats=True)

            # Losses
            distill_loss  = F.kl_div(F.log_softmax(student_logits, dim=-1), F.softmax(teacher_logits, dim=-1), reduction='batchmean')
            contrast_loss = get_contrastive_loss(t_feats, i_feats, student_model.module.losses.logit_scale_contrast)
            id_loss = student_model.module.losses.get_id_loss(t_feats, i_feats, person_ids)
            loss = 3 * distill_loss + 1 * contrast_loss + 3 * id_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if dist.get_rank()==0:
            print(f"[Epoch {epoch+1}] Avg Loss {total_loss/len(train_loader):.4f}")
            wandb.log({"epoch_loss": total_loss/len(train_loader),
                       "distill_loss": distill_loss.item(),
                       "contrast_loss": contrast_loss.item(),
                       "id_loss": id_loss.item(),
                       "lr": scheduler.get_last_lr()[0]})
            if (epoch+1) >= args.eval_start:
                student_model.eval()
                sims_matrix = evaluate(student_model.module.student, test_loader, SimpleTokenizer(),
                                       device, student_model.module.losses.logit_scale_contrast)
                eval_result = itm_eval(sims_matrix, test_dataset.img2person, test_dataset.txt2person, eval_mAP=True)
                wandb.log(eval_result)
                print("Eval:", eval_result)
        dist.barrier()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_annotation_file",type=list, default=["/home/sooyoung/ps/CUHK-PEDES/processed_data/train.json"])
    parser.add_argument("--test_annotation_file", type=str,  default="/home/sooyoung/ps/CUHK-PEDES/processed_data/test.json")
    parser.add_argument("--config_path",          type=str,  default="/home/sooyoung/ps/BLIP/configs/retrieval_CUHK.yaml")
    parser.add_argument("--image_root",     type=str, default="/home/sooyoung/ps/CUHK-PEDES/imgs")
    parser.add_argument("--embedding_path", type=str, default="/home/sooyoung/ps/BLIP/BLIP_top1_78.4_pre-computed_embedding")
    parser.add_argument("--batch_size",     type=int, default=64)
    parser.add_argument("--num_workers",    type=int, default=4)
    parser.add_argument("--checkpoint_path", type=str, default="/home/sooyoung/ps/BLIP/checkpoint_best78.4.pth")
    parser.add_argument("--epochs",    type=int, default=60)
    parser.add_argument("--lr",        type=float, default=1e-5)
    parser.add_argument("--scheduler", type=str,   default="cosine")
    parser.add_argument("--eval_start",type=int,   default=1)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--exp_name",    type=str, required=True)
    parser.add_argument("--wandb_notes", type=str, required=True)
    parser.add_argument("--use_id_loss", action="store_true")
    args = parser.parse_args()
    config = yaml.load(open(args.config_path,'r'), Loader=yaml.FullLoader)
    main(args, config)