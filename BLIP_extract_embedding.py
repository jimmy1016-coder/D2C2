import torch
import torch.nn as nn
import numpy as np
import os
import json
import yaml
import torch.nn.functional as F

from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader

from models.blip_retrieval import BLIP_Retrieval
from data.flickr30k_dataset import ps_dataset_generate_embedding

train_ann_file = ["/home/sooyoung/ps/CUHK-PEDES/processed_data/train.json"]
img_root = "/home/sooyoung/ps/CUHK-PEDES/imgs"
image_res = 384
checkpoint_path = "./checkpoint_best78.4.pth"
config_path = "/home/sooyoung/ps/BLIP/configs/retrieval_CUHK.yaml"
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
print("Config loaded successfully")

output_dir = "./BLIP_top1_78.4_pre-computed_embedding_for_test"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Output directory created: {output_dir}")
else:
    print(f"Output directory already exists: {output_dir}")

normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                 std=(0.26862954, 0.26130258, 0.27577711))
train_transform = transforms.Compose([
    transforms.Resize((image_res, image_res), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    normalize,
])

ps_train_dataset = ps_dataset_generate_embedding(ann_file=train_ann_file, 
                                                transform=train_transform, 
                                                image_root=img_root)
train_loader = DataLoader(dataset=ps_train_dataset, 
                          batch_size=1, shuffle=True, 
                          num_workers=4)

model = BLIP_Retrieval(image_size=config['image_res'], vit=config['vit'], 
                    vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                    queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])

checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint['model']
msg = model.load_state_dict(state_dict, strict=False)
print('load checkpoint from %s' % checkpoint_path)
print(msg)

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

for batch in tqdm(train_loader):
    image, text, index = batch
    
    with torch.no_grad():
        model.temp.clamp_(0.001, 0.5)
        
    text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
    text_id    = text_input.input_ids
    text_attention = text_input.attention_mask
    
    text_output = model.text_encoder(text_id, attention_mask = text_attention, mode='text')  
    text_embed  = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
    
    image = image.to(device)
    image_feat = model.visual_encoder(image)
    
    image_embed = model.vision_proj(image_feat[:,0,:])            
    image_embed = F.normalize(image_embed,dim=-1)      
        

    torch.save({
        "text_id": text_id,
        "text_attention": text_attention,
        "text_embed": text_embed,
        "image_feat": image_feat,
        "image_embed": image_embed
    }, os.path.join(output_dir, f"{index.item()}.pt"))
