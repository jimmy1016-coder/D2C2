'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.blip_retrieval import blip_retrieval
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader


def train(model, data_loader, optimizer, epoch, device, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i,(image, caption, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device,non_blocking=True)   
        idx = idx.to(device,non_blocking=True)   
       
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))

        loss_ita, loss_itm = model(image, caption, alpha=alpha, idx=idx)                  
        loss = loss_ita + loss_itm
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


# @torch.no_grad()
# def evaluation(model, data_loader, device, config):
#     # test
#     model.eval() 
    
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Evaluation:'    
    
#     print('Computing features for evaluation...')
#     start_time = time.time()  

#     texts = data_loader.dataset.text   
#     num_text = len(texts)
#     text_bs = 256
#     text_ids = []
#     text_embeds = []  
#     text_atts = []
#     for i in range(0, num_text, text_bs):
#         text = texts[i: min(num_text, i+text_bs)]
#         text_input  = model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device) 
#         text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
#         text_embed  = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
#         text_embeds.append(text_embed)   
#         text_ids.append(text_input.input_ids)
#         text_atts.append(text_input.attention_mask)
    
#     text_embeds = torch.cat(text_embeds,dim=0)
#     text_ids    = torch.cat(text_ids,dim=0)
#     text_atts   = torch.cat(text_atts,dim=0)
#     text_ids[:,0] = model.tokenizer.enc_token_id
    
#     image_feats = []
#     image_embeds = []
#     for image, img_id in data_loader: 
#         image = image.to(device) 
#         image_feat = model.visual_encoder(image)   
#         image_embed = model.vision_proj(image_feat[:,0,:])            
#         image_embed = F.normalize(image_embed,dim=-1)      
        
#         image_feats.append(image_feat.cpu())
#         image_embeds.append(image_embed)
     
#     image_feats = torch.cat(image_feats,dim=0)
#     image_embeds = torch.cat(image_embeds,dim=0)
    
#     sims_matrix = image_embeds @ text_embeds.t()
#     # score_matrix_i2t = torch.full((len(data_loader.dataset.image),len(texts)),-100.0).to(device)
    
#     num_tasks = utils.get_world_size()
#     rank = utils.get_rank() 

#     sims_matrix = sims_matrix.t()
#     score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)

#     step = sims_matrix.size(0)//num_tasks + 1
    
#     start = rank*step
#     end = min(sims_matrix.size(0),start+step)    
    
#     for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        
#         topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
#         topk_idx = topk_idx.cpu()
#         encoder_output = image_feats[topk_idx].to(device)
#         encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
#         output = model.text_encoder(text_ids[start+i].repeat(config['k_test'],1), 
#                                     attention_mask = text_atts[start+i].repeat(config['k_test'],1),
#                                     encoder_hidden_states = encoder_output,
#                                     encoder_attention_mask = encoder_att,                             
#                                     return_dict = True,
#                                    )
#         score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
#         score_matrix_t2i[start+i,topk_idx] = score + topk_sim

#     if args.distributed:
#         dist.barrier()   
#         #torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
#         torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        
        
#     total_time = time.time() - start_time
#     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#     print('Evaluation time {}'.format(total_time_str)) 

#     return  score_matrix_t2i.cpu().numpy()

@torch.no_grad()
def evaluation(model, data_loader, device, config, args):
    import torch.distributed as dist
    
    model.eval() 
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    start_time = time.time()  

    # --- Text encoding ---
    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_ids, text_embeds, text_atts = [], [], []

    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input  = model.tokenizer(text, padding='max_length', truncation=True,
                                      max_length=35, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids,
                                         attention_mask=text_input.attention_mask,
                                         mode='text')  
        text_embed  = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
        text_embeds.append(text_embed)   
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)
    
    text_embeds = torch.cat(text_embeds,dim=0)
    text_ids    = torch.cat(text_ids,dim=0)
    text_atts   = torch.cat(text_atts,dim=0)
    text_ids[:,0] = model.tokenizer.enc_token_id
    
    # --- Image encoding ---
    image_feats, image_embeds = [], []
    for image, img_id in data_loader: 
        image = image.to(device) 
        image_feat = model.visual_encoder(image)   
        image_embed = model.vision_proj(image_feat[:,0,:])            
        image_embed = F.normalize(image_embed,dim=-1)      
        
        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)
     
    image_feats = torch.cat(image_feats,dim=0)
    image_embeds = torch.cat(image_embeds,dim=0)
    
    sims_matrix = (image_embeds @ text_embeds.t()).t()

    # --- 분산 처리 준비 ---
    world_size = utils.get_world_size()
    rank = utils.get_rank()

    score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)

    # torch.chunk를 이용해 rank별 row 분할
    chunks = torch.chunk(torch.arange(sims_matrix.size(0)), world_size)
    my_rows = chunks[rank]

    for local_i, idx in enumerate(metric_logger.log_every(my_rows, 50, header)):
        sims = sims_matrix[idx]

        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        topk_idx = topk_idx.cpu()

        encoder_output = image_feats[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)

        output = model.text_encoder(text_ids[idx].repeat(config['k_test'],1), 
                                    attention_mask = text_atts[idx].repeat(config['k_test'],1),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,
                                    return_dict = True)
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[idx, topk_idx] = score + topk_sim

    # --- 모든 rank 동기화 및 합치기 ---
    if args.distributed:
        dist.barrier()   # 모든 프로세스 대기
        dist.all_reduce(score_matrix_t2i, op=dist.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if rank == 0:
        print('Evaluation time {}'.format(total_time_str)) 
        return score_matrix_t2i.cpu().numpy()
    else:
        return None

    
@torch.no_grad()
def itm_eval(scores_t2i, img2person, txt2person, eval_mAP):
    scores_t2i = torch.tensor(scores_t2i)
    img2person = torch.tensor(img2person)
    txt2person = torch.tensor(txt2person)
    index = torch.argsort(scores_t2i, dim=-1, descending=True)
    pred_person = img2person[index]
    matches = (txt2person.view(-1, 1).eq(pred_person)).long()

    def acc_k(matches, k=1):
        matches_k = matches[:, :k].sum(dim=-1)
        matches_k = torch.sum((matches_k > 0))
        return 100.0 * matches_k / matches.size(0)

    # Compute metrics
    ir1 = acc_k(matches, k=1).item()
    ir5 = acc_k(matches, k=5).item()
    ir10 = acc_k(matches, k=10).item()
    ir_mean = (ir1 + ir5 + ir10) / 3

    if eval_mAP:
        real_num = matches.sum(dim=-1)
        tmp_cmc = matches.cumsum(dim=-1).float()
        order = torch.arange(start=1, end=matches.size(1) + 1, dtype=torch.long)
        tmp_cmc /= order
        tmp_cmc *= matches
        AP = tmp_cmc.sum(dim=-1) / real_num
        mAP = AP.mean() * 100.0
        eval_result = {'r1': ir1,
                       'r5': ir5,
                       'r10': ir10,
                       'r_mean': ir_mean,
                       'mAP': mAP.item()
                       }
    else:
        eval_result = {'r1': ir1,
                    'r5': ir5,
                    'r10': ir10,
                    'r_mean': ir_mean,}
    return eval_result



def main(args, config):
    utils.init_distributed_mode(args)    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('retrieval_%s'%config['dataset'], config)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False], 
                                                          collate_fns=[None,None,None])   
   

    #### Model #### 
    print("Creating model")
    model = blip_retrieval(pretrained=config['pretrained'], image_size=config['image_res'], vit=config['vit'], 
                            vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                            queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])

    model = model.to(device)   
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module   

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay']) 
    
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()    

    for epoch in range(0, config['max_epoch']):    
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            train_stats = train(model, train_loader, optimizer, epoch, device, config)     
            #score_test_t2i = evaluation(model_without_ddp, test_loader, device, config)        
            score_test_t2i = evaluation(model_without_ddp, test_loader, device, config, args)        
        
        if args.distributed:
            dist.barrier()            
    
        if utils.is_main_process():  
      
            #val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)  
            test_result = itm_eval(score_test_t2i, test_loader.dataset.img2person, test_loader.dataset.txt2person, eval_mAP=True)  
                
            print(test_result)
                                
            if test_result['r1']>best:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))  
                best = test_result['r1']        
                best_epoch = epoch  
            
            if args.evaluate:                
                val_result = {}
                log_stats = {**{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},                  
                            }
                with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")     
            else:
              
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},  
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")   
                    
        if args.evaluate: 
            break

        dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/retrieval_CUHK.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_CUHK')        
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()
    yaml2 = yaml.YAML(typ='rt')
    config = yaml2.load(open(args.config, 'r'))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml2.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)