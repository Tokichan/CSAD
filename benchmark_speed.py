#!/usr/bin/python
# -*- coding: utf-8 -*-
from time import time
import numpy as np
import torch.cuda
from torch import nn
import tqdm
import timm
from torch.nn import functional as F

from models.segmentation.model import Segmentor
from models.model import LocalStudent, AutoEncoder
from scipy.spatial.distance import mahalanobis
from export_model import FixedPatchClassDetector

# without encoder
class ResNetTeacher(nn.Module):
    def __init__(self,out_dim=512,feat_size=64):
        super(ResNetTeacher, self).__init__()
        self.out_dim = out_dim
        self.feat_size = feat_size
        self.proj = nn.Conv2d(1024+512, self.out_dim, kernel_size=1, stride=1, padding=0)
        self.proj.requires_grad_(False)
        
    def forward(self, x):
        x = [x[1],x[2]]
        concat_feat = []
        for i in range(len(x)):
            feat = x[i]
            feat = F.interpolate(feat, size=self.feat_size, mode='bilinear',align_corners=False)
            concat_feat.append(feat)
        concat_feat = torch.cat(concat_feat,dim=1)
        concat_feat = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(concat_feat)
        proj_feat = self.proj(concat_feat)
        return proj_feat


def benchmark(model,size,num=1000,bs=8):
    # Half iterations used to warm up
    times = []
    with torch.no_grad():
        for i in tqdm.tqdm(range(num)):
            data = torch.randn(bs,3,size,size).cuda()
            t = time()
            model(data)
            times.append(time()-t)
    print("-------------Result---------------")
    print(f"Batch size: {bs}")
    print(f"Average latency: {np.mean(times[-num//2:])*1000:.2f}ms")
    print(f"Throughput: {(bs*(num//2))/(np.sum(times[-num//2:])):.2f}fps")
    print(f"Memory Usage:{(torch.cuda.memory_stats()['reserved_bytes.all.peak']/1024**2)}MB")
    
if __name__ == "__main__":
    encoder = timm.create_model(
        'hf_hub:timm/wide_resnet50_2.tv2_in1k',
        pretrained=True,
        features_only=True,
        out_indices=[1,2,3]
    ).cuda().eval()

    segmentor = Segmentor(num_classes=5,in_dim=[256,1024],in_size=256).cuda().eval()
    teacher = ResNetTeacher(out_dim=512,feat_size=56).cuda().eval()

    local_student = LocalStudent(
        feat_size=56,
        out_dim=512,
        padding=False
    ).cuda().eval()

    global_student = AutoEncoder(
        out_size=56,
        out_dim=512,
        base_dim=64,
        input_size=256
    ).cuda().eval()

    patch_hist = FixedPatchClassDetector(num_classes=5,segmap_size=256,use_nahalanobis=True).cuda()

    local_q_start = torch.e
    local_q_end = torch.pi
    global_q_start = torch.e
    global_q_end = torch.pi

    patch_hist_mean = torch.e
    patch_hist_std = torch.pi
    LGST_mean = torch.e
    LGST_std = torch.pi

    def inference(image):
        feat = encoder(image)

        # Patch Histogram branch
        segmap = segmentor(feat)
        segmap = torch.argmax(segmap,dim=1)
        patch_hist_score = patch_hist.detect_grid(segmap)

        # LGST branch
        teacher_feat = teacher(feat)
        local_feat,global_feat = local_student(image)
        ae_global_feat = global_student(image)

        diff_local = torch.pow(teacher_feat-local_feat,2).mean(1)
        diff_global = torch.pow(global_feat-ae_global_feat,2).mean(1)

        # map normalization
        diff_local = 0.1 * (diff_local - local_q_start) / (local_q_end - local_q_start)
        diff_global = 0.1 * (diff_global - global_q_start) / (global_q_end - global_q_start)

        LGST_score = torch.max(diff_local+diff_global)

        # score fusion
        patch_hist_score = (patch_hist_score-patch_hist_mean)/patch_hist_std
        LGST_score = (LGST_score-LGST_mean)/LGST_std

        return patch_hist_score+LGST_score

    benchmark(inference,256,1000,8)
    # benchmark(inference,256,1000,1)

