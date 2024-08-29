import torch
import numpy as np
import os
import timm
from models.segmentation.segmentor import SegmentorTrainer
from models.segmentation.dataloader import SegmentDataset,SegmentDatasetTest, Padding2Resize
from torch.utils.data import DataLoader
import yaml
import sys
import random
sys.path.append(".")

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    for image_size in [256]:
        categories = ["breakfast_box","juice_bottle","pushpins","screw_bag","splicing_connectors",]#
        for category in categories:
            seed_everything(42)
            config = read_config(f"{project_root}/configs/segmentor/{category}.yaml")
            config['image_size'] = image_size
            config['train_image_path'] = config['train_image_path'].replace("category",category).replace("PROJECT_ROOT",project_root)
            config['test_image_path'] = config['test_image_path'].replace("category",category).replace("PROJECT_ROOT",project_root)
            config['val_image_path'] = config['val_image_path'].replace("category",category).replace("PROJECT_ROOT",project_root)
            config['mask_root'] = config['mask_root'].replace("category",category).replace("PROJECT_ROOT",project_root)
            config['model_path'] = config['model_path'].replace("category",category).replace("PROJECT_ROOT",project_root)
            config['model_path'] = config['model_path'].replace(".pth",f"_{config['image_size']}.pth")
            config['category'] = category
            
            sup_dataset = SegmentDataset(
                image_path=config['train_image_path'],
                sup=True,
                mask_root=config['mask_root'],
                image_size=config['image_size'],
                config=config
            )
            config['pad2resize'] = sup_dataset.pad2resize
            sup_dataloader = DataLoader(sup_dataset,
                                        batch_size=4,
                                        shuffle=True,
                                        drop_last=True,
                                        pin_memory=False,
                                        num_workers=0)
            unsup_dataset = SegmentDataset(
                image_path=config['train_image_path'],
                sup=False,
                mask_root=config['mask_root'],
                image_size=config['image_size'],
                config=config
            )
            unsup_dataloader = DataLoader(unsup_dataset,
                                        batch_size=4,
                                        shuffle=True,
                                        drop_last=True,
                                        pin_memory=False,
                                        num_workers=0)
            test_dataset = SegmentDatasetTest(
                image_path=config['test_image_path'],
                image_size=config['image_size']
            )
            test_dataloader = DataLoader(test_dataset,batch_size=1,shuffle=False)
            full_train_dataset = SegmentDatasetTest(
                image_path=config['train_image_path'],
                image_size=config['image_size']
            )
            full_train_dataloader = DataLoader(full_train_dataset,batch_size=1,shuffle=False)
            val_dataset = SegmentDatasetTest(
                image_path=config['val_image_path'],
                image_size=config['image_size']
            )
            val_dataloader = DataLoader(val_dataset,batch_size=1,shuffle=False)


            encoder = timm.create_model('hf_hub:timm/wide_resnet50_2.tv2_in1k'
                                                ,pretrained=True,
                                                features_only=True,
                                                out_indices=[1,2,3])
            for name,param in encoder.named_parameters():
                param.requires_grad = False
            segmentor = SegmentorTrainer(encoder,config)
            segmentor.fit(sup_dataloader,unsup_dataloader,val_dataloader,test_dataloader,full_train_dataloader)


            print("Done!")