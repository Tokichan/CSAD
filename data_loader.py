import numpy as np
from torch.utils.data import Dataset
import torch
import glob
import torch
import random
from datasets import load_dataset as hf_load_dataset
from torchvision import transforms
import os
from PIL import Image
from models.segmentation.dataloader import get_padding_functions


def syn_shuffle(lst0,lst1,lst2,lst3):
    lst = list(zip(lst0,lst1,lst2,lst3))
    random.shuffle(lst)
    lst0,lst1,lst2,lst3 = zip(*lst)
    return lst0,lst1,lst2,lst3

def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)


class MVTecLOCODataset(Dataset):

    def __init__(self, root, image_size, phase,category,use_pad=True,config=None):
        self.phase=phase
        self.category = category
        self.image_size = image_size
        
        self.use_pad = use_pad
        self.build_transform()
        

        if phase=='train':
            print(f"Loading MVTec LOCO {self.category} (train)")
            self.img_path = os.path.join(root,category, 'train')
        elif phase=='eval':
            print(f"Loading MVTec LOCO {self.category} (validation)")
            self.img_path = os.path.join(root,category, 'validation')
        else:
            print(f"Loading MVTec LOCO {self.category} (test)")
            self.img_path = os.path.join(root,category, 'test')
            self.gt_path = os.path.join(root,category, 'ground_truth')
        assert os.path.isdir(os.path.join(root,category)), 'Error MVTecLOCODataset category:{}'.format(category)

        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset() # self.labels => good : 0, anomaly : 1
        
        # load dataset
        self.load_to_gpu()

    def build_transform(self):
        self.norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.resize_norm_transform = transforms.Compose([
            transforms.Resize((self.image_size,self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.aug_tranform = transforms.RandomChoice([
            transforms.ColorJitter(brightness=0.2),
            transforms.ColorJitter(contrast=0.2),
            transforms.ColorJitter(saturation=0.2),
        ])
        self.transform_gt = transforms.Compose([
            transforms.ToTensor(),
        ])

        

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        
        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0]*len(img_paths))
                tot_labels.extend([0]*len(img_paths))
                tot_types.extend(['good']*len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*")
                gt_paths = [g for g in gt_paths if os.path.isdir(g)]
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                if len(gt_paths)==0:
                    gt_paths = [0]*len(img_paths)
                
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1]*len(img_paths))
                tot_types.extend([defect_type]*len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types


    def __len__(self):
        return len(self.img_paths)
    
    def load_to_gpu(self):
        
        self.pad_func, self.pad2resize = get_padding_functions(Image.open(self.img_paths[0]).size,target_size=self.image_size)
        self.pad_func_linear, self.pad2resize_linear = get_padding_functions(Image.open(self.img_paths[0]).size,target_size=self.image_size,mode='bilinear')
        
        self.samples = list()
        self.images = list()
        for i in range(len(self.img_paths)):
            img_path, gt, label, img_type = self.img_paths[i], self.gt_paths[i], self.labels[i], self.types[i]
            img = Image.open(img_path).convert('RGB')
            
            self.images.append(img.copy())

            
            resize_img = self.resize_norm_transform(img)
            pad_img = self.norm_transform(self.pad_func(img))
            

            self.samples.append({
                'image': resize_img.cuda(),
                'pad_image': pad_img.cuda(),
                'label': label,
                'name': os.path.basename(img_path[:-4]),
                'type': img_type,
                'path': img_path,
            })

    def __getitem__(self, idx):
        

        if self.phase == 'train':
            # augmentation
            
            aug_image = self.aug_tranform(self.images[idx])
            self.samples[idx]['aug_image'] = self.resize_norm_transform(aug_image).cuda()

        self.samples[idx]['idx'] = torch.tensor([idx])
        
        return self.samples[idx]

    

class ImageNetDataset(Dataset):
    def __init__(self,transform=None,):
        super().__init__()
        print("Loading ImageNet")
        self.dataset = hf_load_dataset('Maysee/tiny-imagenet', split='train')
        # self.dataset = self.dataset['train']
        # to tensor and apply transform
        self.transform = transform if transform is not None else transforms.ToTensor()
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"].convert("RGB")
        image = self.transform(image).cuda()
        # print(image.shape)
        return image
    



