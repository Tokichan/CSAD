from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchvision.transforms import functional as F
import numpy as np
import cv2
from PIL import Image
import tqdm
from torchvision import transforms
import glob
from scipy.ndimage import binary_fill_holes
from rich.progress import track
from models.segmentation.lsa import LabeledLSA
import albumentations as A

def merge_masks(masks):
    # remove empty masks
    new_mask = list()
    for i,mask in enumerate(masks):
        if np.sum(mask) > 0:
            new_mask.append(mask)
    masks = new_mask


    result_mask = np.zeros_like(masks[0],dtype=np.uint8)
    sorted_masks = sorted(masks,key=lambda x:np.sum(x),reverse=True)
    mask_sum = np.array([np.sum(mask) for mask in masks])
    mask_order = np.argsort(mask_sum)[::-1]
    mask_map = {order+1:i+1 for i,order in enumerate(mask_order)}
    mask_map[0] = 0
    for i,mask in enumerate(sorted_masks):
        result_mask[mask!=0] = np.ones_like(mask)[mask!=0]*(i+1)

    new_mask = np.zeros_like(result_mask)
    for i, order in enumerate(mask_order+1):
        new_mask[result_mask==order] = mask_map[order]
    return new_mask

def split_masks_from_one_mask(masks):
    result_masks = list()
    for i in range(1,np.max(masks)+1):
        mask = np.zeros_like(masks)
        mask[masks==i] = 255
        #print(np.sum(mask>0))
        if np.sum(mask!=0) > 100:
            result_masks.append(mask)
    return result_masks
class Padding2Resize():
    def __init__(self, pad_l, pad_t, pad_r, pad_b):
        self.pad_l = pad_l
        self.pad_t = pad_t
        self.pad_r = pad_r
        self.pad_b = pad_b

    def __call__(self,image,target_size,mode='nearest'):
        shape = len(image.shape)
        if shape == 3:
            image = image[None,:,:,:]
        elif shape == 2:
            image = image[None,None,:,:]
        # B,C,H,W
        if self.pad_b == 0:
            image = image[:,:,self.pad_t:]
        else:
            image = image[:,:,self.pad_t:-self.pad_b]
        if self.pad_r == 0:
            image = image[:,:,:,self.pad_l:]
        else:
            image = image[:,:,:,self.pad_l:-self.pad_r]
        
        if isinstance(image,np.ndarray):
            image = cv2.resize(image,(target_size,target_size),interpolation=cv2.INTER_NEAREST if mode == 'nearest' else cv2.INTER_LINEAR)
        elif isinstance(image,torch.Tensor):
            image = torch.nn.functional.interpolate(image, size=(target_size,target_size), mode=mode)


        if shape == 3:
            return image[0]
        elif shape == 2:
            return image[0,0]
        return image
    
def get_padding_functions(orig_size,target_size=256,resize_target_size=None,mode='nearest',fill=0):
    """
        padding_func, inverse_padding_func = get_padding_functions(image.size,target_size=256)
        image2 = padding_func(image) # image2.size = (256,256) with padding
        image2.show()
        image3 = inverse_padding_func(image2) # image3.size = (256,256) without padding
        image3.show()
    """
    resize_target_size = target_size if resize_target_size is None else resize_target_size
    imsize = orig_size
    long_size = max(imsize)
    scale = target_size / long_size
    new_h = int(imsize[1] * scale + 0.5)
    new_w = int(imsize[0] * scale + 0.5)

    if (target_size - new_w) % 2 == 0:
        pad_l = pad_r = (target_size - new_w) // 2
    else:
        pad_l,pad_r = (target_size - new_w) // 2,(target_size - new_w) // 2 + 1
    if (target_size - new_h) % 2 == 0:
        pad_t = pad_b = (target_size - new_h) // 2
    else:
        pad_t,pad_b = (target_size - new_h) // 2,(target_size - new_h) // 2 + 1
    inter =  Image.NEAREST if mode == 'nearest' else Image.BILINEAR

    padding_func = transforms.Compose([
        transforms.Resize((new_h,new_w),interpolation=inter),
        transforms.Pad((pad_l, pad_t, pad_r, pad_b), fill=fill, padding_mode='constant')
    ])
    return padding_func, Padding2Resize(pad_l,pad_t,pad_r,pad_b)

class SegmentDataset(Dataset):
    def __init__(self,image_path,mask_root,sup=True,image_size=256,use_padding=False,config=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.use_padding = use_padding
        self.sup = sup
        # print(f"Building {'supervise' if sup else 'unsupervise'} dataset...")
        self.image_paths = glob.glob(image_path)
        self.info_path = mask_root+"/info/"
        self.gt_paths = glob.glob(mask_root+"/*/filtered_cluster_map.png")
        self.segment_paths = glob.glob(mask_root+"/*/refined_masks.png")
        self.background_paths = glob.glob(mask_root+"/*/background.jpg")
        self.norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        


        self.image_dataset = self.image_paths
        self.padding_func, self.pad2resize = get_padding_functions(Image.open(self.image_dataset[0]).size,target_size=image_size)
        self.background_padding_func, self.background_pad2resize = get_padding_functions(Image.open(self.background_paths[0]).size,target_size=image_size,fill=255)
        
        if not self.use_padding:
            # use resize
            self.padding_func_nearest = transforms.Resize((image_size,image_size),interpolation=Image.NEAREST)
            self.padding_func_linear = transforms.Resize((image_size,image_size),interpolation=Image.BILINEAR)
            
            self.background_padding_func = transforms.Resize((image_size,image_size),interpolation=Image.NEAREST)
        
        self.good_indices = np.sort(np.load(self.info_path+"filtered_histogram_indices.npy")).tolist()
        self.is_good_indices = np.array([1 if i in self.good_indices else 0 for i in range(len(self.image_paths))])
        
        

        if self.sup:
            self.image_paths = [image_path for i,image_path in enumerate(self.image_paths) if i in self.good_indices]
            self.gt_paths = [gt_path for i,gt_path in enumerate(self.gt_paths) if i in self.good_indices]
        else:
            self.image_paths = [image_path for i,image_path in enumerate(self.image_paths) if i not in self.good_indices]
            self.gt_paths = [gt_path for i,gt_path in enumerate(self.gt_paths) if i not in self.good_indices]
        
        
        self.image_dataset = []
        self.background_dataset = []
        self.gt_dataset = []
        self.transform = self.create_transform()
        
        for i, image_path in tqdm.tqdm(list(enumerate(self.image_paths)),desc=f'loading {"sup" if sup else "unsup"} datas...'):
            image = self.padding_func_linear(Image.open(image_path).convert("RGB"))
            if self.config['fill_holes']:
                gt = np.array(Image.open(self.gt_paths[i]).convert("L"))
                gt = split_masks_from_one_mask(gt)
                gt = merge_masks([binary_fill_holes(mask) if self.config['fill_holes'] else mask for mask in gt])
                gt = self.padding_func_nearest(Image.fromarray(gt))
            else:
                gt = self.padding_func_nearest(Image.open(self.gt_paths[i]).convert("L"))
            self.image_dataset.append(np.array(image))
            self.gt_dataset.append(np.array(gt))
            
        ########################
        # prepare for LSA
        if self.sup:
            # self.lsa_images = [np.array(padding_func(Image.open(image_path).convert("RGB"))) for i,image_path in enumerate(self.image_dataset) if i in self.good_indices]
            self.lsa_images = self.image_dataset
            self.lsa_backgrounds = [np.array(self.background_padding_func(Image.open(image_path).convert("L"))) for i,image_path in enumerate(self.background_paths) if i in self.good_indices]
            # self.lsa_labels = [np.array(padding_func(Image.open(image_path).convert("L"))) for i,image_path in enumerate(self.gt_paths) if i in self.good_indices]
            self.lsa_labels =  self.gt_dataset
            self.lsa_masks = [np.array(self.padding_func_nearest(Image.open(image_path).convert("L"))) for i,image_path in  enumerate(self.segment_paths) if i in self.good_indices]
            self.lsa_ratio = self.config['LSA_ratio']
            self.lsa = LabeledLSA(
                self.lsa_images,
                self.lsa_masks,
                self.lsa_labels,
                self.lsa_backgrounds,
                self.config['LSA_config']
            )
        ########################
        
        

    def create_transform(self):
        return A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.01,
                scale_limit=0.01,
                rotate_limit=2,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.05,
                contrast_limit=0.05,
                p=0.5
            ),
        ])
    
    def apply_transform(self,image,mask,transform):
        transformed = transform(image=image,mask=mask)
        return transformed['image'],transformed['mask']



    def __len__(self):
        return len(self.image_dataset)
    
    def __getitem__(self, index):
        image = self.image_dataset[index]
        gt = self.gt_dataset[index]
        gt_path = self.gt_paths[index]
        rand_gt = self.gt_dataset[np.random.randint(len(self.gt_dataset))]


        if self.sup:
            # only augment in supervised mode
            if np.random.rand() > 1.0 - self.lsa_ratio:
                image, gt = self.lsa.augment(index)
            image, gt = self.apply_transform(image,gt,self.transform)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # cv2.imshow("image",image)
            # cv2.waitKey(0)

        image = self.norm_transform(image)
        gt = torch.unsqueeze(torch.from_numpy(gt),dim=0).type(torch.long)
        rand_gt = torch.unsqueeze(torch.from_numpy(rand_gt),dim=0).type(torch.long)

        image = image.to(self.device)
        gt = gt.to(self.device)
        rand_gt = rand_gt.to(self.device)

        return image, gt, rand_gt, gt_path
    
    
class SegmentDatasetTest(Dataset):
    def __init__(self,image_path,image_size=256,use_padding=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = transforms.Compose([
            #transforms.Resize((image_size,image_size),interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_paths = glob.glob(image_path)
        if use_padding:
            padding_func = get_padding_functions(Image.open(self.image_paths[0]).size,target_size=image_size)
        else:
            padding_func = transforms.Resize((image_size,image_size),interpolation=Image.BILINEAR)
        self.image_dataset = [transform(padding_func(Image.open(image_path).convert("RGB"))).to(self.device) for image_path in tqdm.tqdm(self.image_paths,desc='loading images...')]
        

    def __len__(self):
        return len(self.image_dataset)
    
    def __getitem__(self, index):
        image = self.image_dataset[index]
        path = self.image_paths[index]
        return image,path
    
def de_normalize(tensor):
    # tensor: (B,C,H,W)
    mean = torch.tensor([0.485, 0.456, 0.406]).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(tensor.device)
    tensor = tensor * std.unsqueeze(0).unsqueeze(2).unsqueeze(3) + mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    return tensor

