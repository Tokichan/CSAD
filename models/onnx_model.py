"""
    This file contains the ONNX/Pytorch model for CSAD.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import timm
from models.model import LocalStudent,AutoEncoder
from models.segmentation.model import Segmentor


class FixedPatchClassDetector(nn.Module):
    """
        Optimized version of patch histogram detector for fast speed
        and support batched input.
        (Predefined patch sizes and overlap ratios)
        (not using EMPatches to extract patches)
    """
    def __init__(self,num_classes=5,segmap_size=256,use_nahalanobis=True):
        super(FixedPatchClassDetector, self).__init__()
        self.num_classes = num_classes
        self.segmap_size = segmap_size
        
        self.use_nahalanobis = use_nahalanobis

        self.hist_mean = torch.nn.Parameter(data=torch.randn((self.num_classes,1)),requires_grad=False)
        self.hist_invcov = torch.nn.Parameter(data=torch.randn((self.num_classes,self.num_classes)),requires_grad=False)
        self.patch_hist_mean = torch.nn.Parameter(data=torch.randn((self.num_classes*4,1)),requires_grad=False)
        self.patch_hist_invcov = torch.nn.Parameter(data=torch.randn((self.num_classes*4,self.num_classes*4)),requires_grad=False)
        self.hist_val_mean = torch.nn.Parameter(data=torch.randn(1)[0],requires_grad=False)
        self.hist_val_std = torch.nn.Parameter(data=torch.randn(1)[0],requires_grad=False)
        self.patch_hist_val_mean = torch.nn.Parameter(data=torch.randn(1)[0],requires_grad=False)
        self.patch_hist_val_std = torch.nn.Parameter(data=torch.randn(1)[0],requires_grad=False)
        

    def set_params(self,params):
        self.hist_mean = torch.nn.Parameter(data=params['hist_mean'].unsqueeze(1),requires_grad=False)
        self.hist_invcov = torch.nn.Parameter(data=params['hist_invcov'],requires_grad=False)
        self.patch_hist_mean = torch.nn.Parameter(data=params['patch_hist_mean'].unsqueeze(1),requires_grad=False)
        self.patch_hist_invcov = torch.nn.Parameter(data=params['patch_hist_invcov'],requires_grad=False)
        self.hist_val_mean = torch.nn.Parameter(data=params['hist_val_mean'],requires_grad=False)
        self.hist_val_std = torch.nn.Parameter(data=params['hist_val_std'],requires_grad=False)
        self.patch_hist_val_mean = torch.nn.Parameter(data=params['patch_hist_val_mean'],requires_grad=False)
        self.patch_hist_val_std = torch.nn.Parameter(data=params['patch_hist_val_std'],requires_grad=False)
    
    @staticmethod
    def mahalanobis(u, v, invcov):
        delta = u - v
        m = torch.matmul(delta.T, torch.matmul(invcov, delta)).squeeze()
        return torch.sqrt(m)

    @staticmethod
    def histogram(seg_map):
        _, max_indices = torch.max(seg_map, dim=1)
        # Create a tensor of zeros with the same shape as the input tensor
        out = torch.zeros_like(seg_map)
        # Set the max values to 1 by using advanced indexing
        out.scatter_(1, max_indices.unsqueeze(1), 1)
        hist = torch.mean(out,dim=[2,3])
        hist = hist.unsqueeze(2)
        return hist
    
    @staticmethod
    def patch_histogram(seg_map):
        # patch_size = 128
        a = seg_map[:,:,:128,:128]
        b = seg_map[:,:,128:,:128]
        c = seg_map[:,:,:128,128:]
        d = seg_map[:,:,128:,128:]
        return torch.concat([
            FixedPatchClassDetector.histogram(a),
            FixedPatchClassDetector.histogram(b),
            FixedPatchClassDetector.histogram(c),
            FixedPatchClassDetector.histogram(d)]
            ,dim=1)



    def forward(self,segmap):
        hist = FixedPatchClassDetector.histogram(segmap)
        patch_hist = FixedPatchClassDetector.patch_histogram(segmap)

        diff_hists = []
        diff_patchhists = []
        for i in range(segmap.shape[0]):
            if self.use_nahalanobis:
                diff_hist = FixedPatchClassDetector.mahalanobis(hist[i],self.hist_mean,self.hist_invcov)
                diff_patchhist = FixedPatchClassDetector.mahalanobis(patch_hist[i],self.patch_hist_mean,self.patch_hist_invcov)
            else:
                diff_hist = ((hist[i]-self.hist_mean)**2).sum()**0.5
                diff_patchhist = ((patch_hist[i]-self.patch_hist_mean)**2).sum()**0.5
            diff_hists.append(diff_hist)
            diff_patchhists.append(diff_patchhist)
        diff_hist = torch.stack(diff_hists)
        diff_patchhist = torch.stack(diff_patchhists)

        diff_hist = (diff_hist-self.hist_val_mean)/self.hist_val_std
        diff_patchhist = (diff_patchhist-self.patch_hist_val_mean)/self.patch_hist_val_std
        
        out_score = diff_hist + diff_patchhist
        return out_score
    
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
    
class CSAD_ONNX(nn.Module):
    def __init__(self,dim=512,num_classes=5):
        super(CSAD_ONNX, self).__init__()
        self.teacher_mean = torch.nn.Parameter(data=torch.randn((1,dim,1,1)),requires_grad=False)
        self.teacher_std = torch.nn.Parameter(data=torch.randn((1,dim,1,1)),requires_grad=False)
        
        self.q_st_start = torch.nn.Parameter(data=torch.randn((1))[0],requires_grad=False)
        self.q_st_end = torch.nn.Parameter(data=torch.randn((1))[0],requires_grad=False)
        self.q_ae_start = torch.nn.Parameter(data=torch.randn((1))[0],requires_grad=False)
        self.q_ae_end = torch.nn.Parameter(data=torch.randn((1))[0],requires_grad=False)

        self.lgst_mean = torch.nn.Parameter(data=torch.randn((1))[0],requires_grad=False)
        self.lgst_std = torch.nn.Parameter(data=torch.randn((1))[0],requires_grad=False)

        self.patch_hist_mean = torch.nn.Parameter(data=torch.randn((1))[0],requires_grad=False)
        self.patch_hist_std = torch.nn.Parameter(data=torch.randn((1))[0],requires_grad=False)

        self.encoder = timm.create_model('hf_hub:timm/wide_resnet50_2.tv2_in1k'
                                                ,pretrained=True,
                                                features_only=True,
                                                out_indices=[1,2,3])
        self.teacher = ResNetTeacher(out_dim=512,feat_size=64)
        self.local_st = LocalStudent(out_dim=512,feat_size=64,padding=True)
        self.autoencoder = AutoEncoder(out_dim=512,out_size=64)
        self.segmentor = Segmentor(num_classes=num_classes)
        self.patch_hist_detector = FixedPatchClassDetector(
            num_classes=num_classes
            ,segmap_size=256,
            use_nahalanobis=True
        )

    def load_model_params(self,models,params):
        self.teacher_mean = torch.nn.Parameter(data=params['teacher_mean'],requires_grad=False)
        self.teacher_std = torch.nn.Parameter(data=params['teacher_std'],requires_grad=False)
        
        self.q_st_start = torch.nn.Parameter(data=params['q_st_start'],requires_grad=False)
        self.q_st_end = torch.nn.Parameter(data=params['q_st_end'],requires_grad=False)
        self.q_ae_start = torch.nn.Parameter(data=params['q_ae_start'],requires_grad=False)
        self.q_ae_end = torch.nn.Parameter(data=params['q_ae_end'],requires_grad=False)

        self.lgst_mean = torch.nn.Parameter(data=params['lgst_mean'],requires_grad=False)
        self.lgst_std = torch.nn.Parameter(data=params['lgst_std'],requires_grad=False)
        if self.lgst_std < 1e-3:
            self.lgst_std = torch.nn.Parameter(data=torch.tensor([1])[0],requires_grad=False)
        
        self.patch_hist_mean = torch.nn.Parameter(data=params['patchhist_mean'],requires_grad=False)
        self.patch_hist_std = torch.nn.Parameter(data=params['patchhist_std'],requires_grad=False)
        if self.patch_hist_std < 1e-3:
            self.patch_hist_std = torch.nn.Parameter(data=torch.tensor([1])[0],requires_grad=False)
        
        self.encoder = models['encoder']
        self.teacher = models['teacher']
        self.local_st = models['local_st']
        self.autoencoder = models['autoencoder']
        self.segmentor = models['segmentor']
        self.patch_hist_detector = models['patch_hist_detector']
        return self


    @classmethod
    def from_pretrained(cls,models,params):
        num_classes = models['segmentor'].fc2.conv3.out_channels
        return cls(num_classes=num_classes).load_model_params(models,params)


    def forward(self, image):
        # shape: (B,3,256,256)
        feat = self.encoder(image)

        # Patch Histogram branch
        segmap = self.segmentor(feat)
        # segmap = torch.argmax(segmap,dim=1)
        patch_hist_score = self.patch_hist_detector(segmap)

        # LGST branch
        teacher_feat = self.teacher(feat)
        teacher_feat = (teacher_feat-self.teacher_mean)/self.teacher_std
        local_feat,global_feat = self.local_st(image)
        ae_global_feat = self.autoencoder(image)

        diff_local = torch.pow(teacher_feat-local_feat,2).mean(1)
        diff_global = torch.pow(global_feat-ae_global_feat,2).mean(1)

        # map normalization
        diff_local = 0.1 * (diff_local - self.q_st_start) / (self.q_st_end - self.q_st_start)
        diff_global = 0.1 * (diff_global - self.q_ae_start) / (self.q_ae_end - self.q_ae_start)

        LGST_score = torch.amax(diff_local+diff_global,dim=(1,2))

        # score fusion
        LGST_score = (LGST_score-self.lgst_mean)/self.lgst_std
        patch_hist_score = (patch_hist_score-self.patch_hist_mean)/self.patch_hist_std



        return LGST_score+patch_hist_score