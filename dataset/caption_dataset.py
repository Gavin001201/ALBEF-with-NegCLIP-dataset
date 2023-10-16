import json
import os
import random
import ast
from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption
import pandas as pd

class re_train_dataset(Dataset):
    def __init__(self, input_filename, transform, image_root, max_words=30):   

        df = pd.read_csv(input_filename, sep="\t", converters={"neg_caption":ast.literal_eval, "neg_image":ast.literal_eval})   
        self.images = df["filepath"].tolist()
        self.captions = df["title"].tolist()
        self.hard_captions = df["neg_caption"].tolist()
        self.hard_images = df["neg_image"].tolist()
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {} 
        self.neg_img_ids = {} 
        n = 0
        for image in self.images:
            img_id = image.split('/')[-1].split('.')[0]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

        print('Done loading data.')        
        # self.ann = []
        # for f in ann_file:
        #     self.ann += json.load(open(f,'r'))
        # self.transform = transform
        # self.image_root = image_root
        # self.max_words = max_words
        # self.img_ids = {}   
        
        # n = 0
        # for ann in self.ann:
        #     img_id = ann['image_id']
        #     if img_id not in self.img_ids.keys():
        #         self.img_ids[img_id] = n
        #         n += 1    

    def __len__(self):
        # return len(self.ann)
        return len(self.captions)
    
    def __getitem__(self, index): 
        path = os.path.join('/mnt/workspace/Project/vision-language-models-are-bows',self.images[index])
        image = Image.open(path).convert('RGB')   
        image = self.transform(image)

        idx = self.img_ids[self.images[index].split('/')[-1].split('.')[0]]

        texts = str(self.captions[index])

        chosen_caption = random.choice(self.hard_captions[index])
        hard_captions = str(chosen_caption)

        chose_image_index = random.choice(self.hard_images[index])

        new_path = os.path.join('/mnt/workspace/Project/vision-language-models-are-bows',self.images[chose_image_index])
        new_images = Image.open(new_path).convert('RGB')
        new_images = self.transform(new_images)

        neg_idx = self.img_ids[self.images[chose_image_index].split('/')[-1].split('.')[0]]
        new_texts = str(self.captions[chose_image_index])

        chosen_caption = random.choice(self.hard_captions[chose_image_index])
        new_hard = str(chosen_caption)
        # 图片， 图片负样本， 文本， 图片负样本对应的文本， 原文本对应的负文本， 图片负样本对应文本对应的负文本
        return image, new_images, texts, new_texts, hard_captions, new_hard, idx, neg_idx
        
        # ann = self.ann[index]
        
        # image_path = os.path.join(self.image_root,ann['image'])        
        # image = Image.open(image_path).convert('RGB')   
        # image = self.transform(image)
        
        # caption = pre_caption(ann['caption'], self.max_words) 

        # return image, caption, self.img_ids[ann['image_id']]
    
    

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index
      
        

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
      
        image = Image.open(ann['image']).convert('RGB')   
        image = self.transform(image)
                
        return image, caption
            

    
