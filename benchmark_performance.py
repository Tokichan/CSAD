#!/usr/bin/python
# -*- coding: utf-8 -*-
from time import time
import numpy as np
import torch.cuda
from torch import nn
from data_loader import MVTecLOCODataset
from torch.utils.data import DataLoader
import sklearn
import tqdm

import matplotlib.pyplot as plt

import os
from itertools import combinations

def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


def de_normalize(tensor):
    # tensor: (B,C,H,W)
    mean = torch.tensor([0.485, 0.456, 0.406]).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(tensor.device)
    tensor = tensor * std.unsqueeze(0).unsqueeze(2).unsqueeze(3) + mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    return tensor

def normalize(score,val_score):
    # max normalization
    if np.max(val_score) != 0:
        score = score/np.max(val_score)
    return score

def normalize2(score,val_score):
    # trimmed z-normalization
    q = np.quantile(val_score,0.2)
    p = np.quantile(val_score,0.8)
    val_score = val_score[(val_score>q) & (val_score<p)]

    score = score-np.mean(val_score)
    if np.std(val_score) != 0:
        score = score/np.std(val_score)
    return score

def normalize3(score,val_score):
    # mean absolute deviation normalization
    avg_dist = np.mean(np.abs(val_score-np.mean(val_score)))
    if avg_dist != 0:
        score = score/avg_dist
    return score


def test_scores(score_name,config):
    experiments = []
    for i in range(1,len(score_name)+1):
        experiments += list(combinations(score_name,i))
    print(experiments)
    true_score_logi, true_score_struc = get_true_score(config)
    score_dict = {}
    for name in score_name:
        raw_score_logi = np.load(f"./anomaly_score/{config['category']}_{name}_logi_score.npy")
        raw_score_struc = np.load(f"./anomaly_score/{config['category']}_{name}_struc_score.npy")
        score_val = np.load(f"./anomaly_score/{config['category']}_{name}_val_score.npy")
        score_logi = normalize2(raw_score_logi,score_val)
        score_struc = normalize2(raw_score_struc,score_val)
        score_dict[name] = (score_logi,score_struc,raw_score_logi,raw_score_struc,score_val)
    with open(f"./score_combine_result.txt","a") as f:
        f.write(f"{str(config['category']).upper()}:\n")
        for experiment in experiments:
            score_logi = np.array([score_dict[name][0] for name in experiment])
            score_struc = np.array([score_dict[name][1] for name in experiment])
            raw_score_logi = np.array([score_dict[name][2] for name in experiment])
            raw_score_struc = np.array([score_dict[name][3] for name in experiment])
            score_val = np.array([score_dict[name][4] for name in experiment])


            score_logi = np.sum(score_logi,axis=0)
            score_struc = np.sum(score_struc,axis=0)
            raw_score_logi = np.sum(raw_score_logi,axis=0)
            raw_score_struc = np.sum(raw_score_struc,axis=0)
            
            auc_logi = sklearn.metrics.roc_auc_score(true_score_logi,score_logi)*100
            auc_struc = sklearn.metrics.roc_auc_score(true_score_struc,score_struc)*100
            f.write(f"{experiment} AUC: {auc_logi:.3f} {auc_struc:.3f}\n")
    
    return auc_logi,auc_struc
            




def get_true_score(config):
    if os.path.exists(f"./anomaly_score/{config['category']}_true_logi_score.npy"):
        true_score_logi = np.load(f"./anomaly_score/{config['category']}_true_logi_score.npy")
        true_score_strc = np.load(f"./anomaly_score/{config['category']}_true_struc_score.npy")
        return true_score_logi, true_score_strc
    else:
        image_size = config['image_size']
    
        dataset_path = config['Datasets']['train']['root']
        test_set = MVTecLOCODataset(
                                    root=dataset_path,
                                    image_size=image_size,
                                    phase='test',
                                    use_pad=True,
                                    category=config['category'])
        test_loader = DataLoader(test_set,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0)
        true_score_logi = []
        true_score_struc = []
        for i,sample in tqdm.tqdm(list(enumerate(test_loader))):
            path = sample['path']
            save_path = path[0].replace("\\","/").replace("mvtec_loco_anomaly_detection","masks")
            anomaly_type = save_path.split("/")[-2]
            image_name = save_path.split("/")[-1].split(".")[0]
            if anomaly_type == "logical_anomalies":
                true_score_logi.append(1)
            elif anomaly_type == "structural_anomalies":
                true_score_struc.append(1)
            elif anomaly_type == "good":
                true_score_logi.append(0)
                true_score_struc.append(0)
        true_score_logi = np.array(true_score_logi)
        true_score_struc = np.array(true_score_struc)
        np.save(f"./anomaly_score/{config['category']}_true_logi_score.npy",true_score_logi)
        np.save(f"./anomaly_score/{config['category']}_true_struc_score.npy",true_score_struc)
        return true_score_logi, true_score_struc

            

if __name__ == "__main__":
    import yaml
    categories =  ["breakfast_box","juice_bottle","pushpins","screw_bag","splicing_connectors"]
    run_name = ["patchhist","LGST"]
    config = read_config("./configs/mvtecloco_train.yaml")
    root = os.path.dirname(os.path.abspath(__file__))
    all_logi_auc = []
    all_struc_auc = []
    for category in categories:
        config = read_config("./configs/mvtecloco_train.yaml")
        config.update({
            "category":category,
            "image_size":256,
            "image_root":f"{root}/datasets/mvtec_loco_anomaly_detection/",
            "mask_root":f"{root}/datasets/masks/{category}/",
            "anomaly_map_path":f"{root}/output/{run_name}/anomaly_maps/mvtec_loco/{category}/test/",
            "segmentor_path":f"{root}/ckpt/segmentor_{category}.pth",
        })
        logi_auc,struc_auc = test_scores(run_name,config)
        all_logi_auc.append(logi_auc)
        all_struc_auc.append(struc_auc)
    print(f"Logical AUC: {np.mean(all_logi_auc):.3f} Structural AUC: {np.mean(all_struc_auc):.3f}")
    print("Average AUC: ",np.mean(all_logi_auc+all_struc_auc))
