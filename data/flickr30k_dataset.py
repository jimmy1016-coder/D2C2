import os
import json
import torch
import random

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from collections import defaultdict
from PIL import Image

from data.utils import pre_caption
from simple_tokenizer import SimpleTokenizer, tokenize

class flickr30k_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        '''        
        url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_train.json'
        filename = 'flickr30k_train.json'

        download_url(url,ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']] # image_id에 해당하는 index
    
class ps_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        anns = []
        for f in ann_file:
            anns += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.person2image = defaultdict(list)
        self.person2text = defaultdict(list)
        person_id2idx = {}
        n = 0
        self.pairs = []
        for ann in anns:
            person_id = ann['id']
            if person_id not in person_id2idx.keys():
                person_id2idx[person_id] = n
                n += 1
            person_idx = person_id2idx[person_id]
            self.person2image[person_idx].append(ann['file_path'])
            for cap in ann['captions']:
                self.pairs.append((ann['file_path'], cap, person_idx))
                self.person2text[person_idx].append(cap)

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        
        image_path, caption, person_idx = self.pairs[index]
        image_path = os.path.join(self.image_root, image_path)
        image = Image.open(image_path).convert('RGB')
        image1 = self.transform(image)
        caption1 = pre_caption(caption, self.max_words)
        return image1, caption1, person_idx



class flickr30k_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json'}
        filenames = {'val':'flickr30k_val.json','test':'flickr30k_test.json'}
        
        download_url(urls[split],ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index    
    
class ps_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.text = []
        self.image = []
        self.txt2person = []
        self.img2person = []
        person2img = defaultdict(list)
        person2txt = defaultdict(list)
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['file_path'])
            person_id = ann['id']
            person2img[person_id].append(img_id)
            self.img2person.append(person_id)
            for caption in ann['captions']:
                self.text.append(pre_caption(caption, self.max_words))
                person2txt[person_id].append(txt_id)
                self.txt2person.append(person_id)
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.ann[index]['file_path'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index


class ps_dataset_generate_embedding(Dataset):
    """
    Embedding generation을 위한 dataset 클래스
    """
    def __init__(self, ann_file, transform, image_root, max_words=30):
        anns = []
        for f in ann_file:
            anns += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.person_idx2image = defaultdict(list)
        self.person_idx2text = defaultdict(list)
        person_id2idx = {}
        n = 0
        self.pairs = []
        for ann in anns:
            person_id = ann['id']
            if person_id not in person_id2idx.keys():
                person_id2idx[person_id] = n
                n += 1
            person_idx = person_id2idx[person_id]
            self.person_idx2image[person_idx].append(ann['file_path'])
            for cap in ann['captions']:
                self.pairs.append((ann['file_path'], cap, person_idx))
                self.person_idx2text[person_idx].append(cap)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, caption, person = self.pairs[index]
        image_path = os.path.join(self.image_root, image_path)
        image = Image.open(image_path).convert('RGB')
        image1 = self.transform(image)
        caption1 = pre_caption(caption, self.max_words)
        return image1, caption1, index



class ps_train_dataset_KD(Dataset):
    """
    Knowledge Distillation을 위한 dataset 클래스
    """
    def __init__(self, ann_file, transform, image_root, embedding_path, max_words=30):
        anns = []
        for f in ann_file:
            anns += json.load(open(f, 'r'))
        self.transform  = transform
        self.image_root = image_root
        self.max_words  = max_words
        self.person_idx2image = defaultdict(list)
        self.person_idx2text  = defaultdict(list)
        self.embedding_path   = embedding_path
        self.clip_tokenizer   = SimpleTokenizer()
        
        person_id2idx = {}
        n = 0
        self.pairs = []
        for ann in anns:
            person_id = ann['id']
            if person_id not in person_id2idx.keys():
                person_id2idx[person_id] = n
                n += 1
            person_idx = person_id2idx[person_id]
            self.person_idx2image[person_idx].append(ann['file_path'])
            for cap in ann['captions']:
                self.pairs.append((ann['file_path'], cap, person_idx))
                self.person_idx2text[person_idx].append(cap)
        self.num_ids = len(person_id2idx)
        
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        
        image_path, caption, person = self.pairs[index]
        embedding_path = os.path.join(self.embedding_path, f"{index}.pt")
        embedding = torch.load(embedding_path, map_location='cpu')
        
        text_id = embedding["text_id"].squeeze(0).detach()
        text_embed = embedding["text_embed"].squeeze(0).detach()
        text_attention = embedding["text_attention"].squeeze(0).detach()
        image_feat = embedding["image_feat"].squeeze(0).detach()
        image_embed = embedding["image_embed"].squeeze(0).detach()
        image_attention = torch.ones(image_feat.size()[:-1], dtype=torch.long).requires_grad_(False)
        
        image_path = os.path.join(self.image_root, image_path)
        image = Image.open(image_path).convert('RGB')
        image1 = self.transform(image)
        caption_ids = tokenize(caption, tokenizer=self.clip_tokenizer, text_length=77, truncate=True)
        _, _ = self._build_random_masked_tokens_and_labels(caption_ids)
        
        return image1, caption_ids, person, text_id, text_attention, image_feat, image_attention, text_embed, image_embed
    
    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.clip_tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.clip_tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15
                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask
                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)
                    # -> rest 10% randomly keep current token
                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)


class ps_eval_dataset_KD(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.text = []
        self.image = []
        self.txt2person = []
        self.img2person = []
        person2img = defaultdict(list)
        person2txt = defaultdict(list)
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['file_path'])
            person_id = ann['id']
            person2img[person_id].append(img_id)
            self.img2person.append(person_id)
            for caption in ann['captions']:
                self.text.append(caption)
                person2txt[person_id].append(txt_id)
                self.txt2person.append(person_id)
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.ann[index]['file_path'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index

