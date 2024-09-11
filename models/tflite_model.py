"""
    This file contains the TFLite model for CSAD.
    remove torch.amax()
    remove dropout layers in AutoEncoder
    use tflite_model_postprocess() to get the output of original CSAD
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import timm
import numpy as np
from models.model import LocalStudent
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
        if not self.use_nahalanobis:
            params['hist_invcov'] = torch.eye(self.num_classes)
        self.hist_invcov = torch.nn.Parameter(data=params['hist_invcov'],requires_grad=False)
        self.patch_hist_mean = torch.nn.Parameter(data=params['patch_hist_mean'].unsqueeze(1),requires_grad=False)
        if not self.use_nahalanobis:
            params['patch_hist_invcov'] = torch.eye(self.num_classes*4)
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
            diff_hist = FixedPatchClassDetector.mahalanobis(hist[i],self.hist_mean,self.hist_invcov)
            diff_patchhist = FixedPatchClassDetector.mahalanobis(patch_hist[i],self.patch_hist_mean,self.patch_hist_invcov)
            diff_hists.append(diff_hist)
            diff_patchhists.append(diff_patchhist)
        diff_hist = torch.stack(diff_hists)
        diff_patchhist = torch.stack(diff_patchhists)

        diff_hist = (diff_hist-self.hist_val_mean)/self.hist_val_std
        diff_patchhist = (diff_patchhist-self.patch_hist_val_mean)/self.patch_hist_val_std
        
        out_score = diff_hist + diff_patchhist
        return out_score

    
class AutoEncoder(nn.Module):
    def __init__(self, out_size=64,out_dim=512,base_dim=64):
        super(AutoEncoder, self).__init__()
        self.out_dim = out_dim
        self.base_dim = base_dim
        self.out_size = out_size

        self.enconv1 = nn.Conv2d(3, self.base_dim//2, kernel_size=4, stride=2, padding=1)
        self.enconv2 = nn.Conv2d(self.base_dim//2, self.base_dim//2, kernel_size=4, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(self.base_dim//2, self.base_dim, kernel_size=4, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=2, padding=1)
        self.enconv5 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=2, padding=1)
        self.enconv6 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=8, stride=1, padding=0)

        self.deconv1 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=1, padding=2)
        self.deconv5 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=1, padding=2)
        self.deconv6 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=4, stride=1, padding=2)
        self.deconv7 = nn.Conv2d(self.base_dim, self.base_dim, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(self.base_dim, self.out_dim, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.2)
        self.dropout6 = nn.Dropout(p=0.2)


    def forward(self, x):
        x1 = F.relu(self.enconv1(x))
        x2 = F.relu(self.enconv2(x1))
        x3 = F.relu(self.enconv3(x2))
        x4 = F.relu(self.enconv4(x3))
        x5 = F.relu(self.enconv5(x4))
        x6 = self.enconv6(x5)

        x = F.interpolate(x6, size=3, mode='bilinear')
        x = F.relu(self.deconv1(x))
        # x = self.dropout1(x)
        x = F.interpolate(x, size=8, mode='bilinear')
        x = F.relu(self.deconv2(x))
        # x = self.dropout2(x)
        x = F.interpolate(x, size=15, mode='bilinear')
        x = F.relu(self.deconv3(x))
        # x = self.dropout3(x)
        x = F.interpolate(x, size=32, mode='bilinear')
        x = F.relu(self.deconv4(x))
        # x = self.dropout4(x)
        x = F.interpolate(x, size=63, mode='bilinear')
        x = F.relu(self.deconv5(x))
        # x = self.dropout5(x)
        x = F.interpolate(x, size=127, mode='bilinear')
        x = F.relu(self.deconv6(x))
        # x = self.dropout6(x)
        x = F.interpolate(x, size=self.out_size, mode='bilinear')
        x = F.relu(self.deconv7(x))
        x = self.deconv8(x)
        return x

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
    
class CSAD_TFLITE(nn.Module):
    def __init__(self,dim=512,num_classes=5):
        super(CSAD_TFLITE, self).__init__()
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
            num_classes=num_classes,
            segmap_size=256,
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


    def forward(self, x):
        # shape: (B,3,256,256)
        feat = self.encoder(x)

        # Patch Histogram branch
        segmap = self.segmentor(feat)
        # segmap = torch.argmax(segmap,dim=1)
        patch_hist_score = self.patch_hist_detector(segmap)

        # LGST branch
        teacher_feat = self.teacher(feat)
        teacher_feat = (teacher_feat-self.teacher_mean)/self.teacher_std
        local_feat,global_feat = self.local_st(x)
        ae_global_feat = self.autoencoder(x)

        diff_local = torch.pow(teacher_feat-local_feat,2).mean(1)
        diff_global = torch.pow(global_feat-ae_global_feat,2).mean(1)

        # map normalization
        diff_local = 0.1 * ((diff_local - self.q_st_start) / (self.q_st_end - self.q_st_start))
        diff_global = 0.1 * ((diff_global - self.q_ae_start) / (self.q_ae_end - self.q_ae_start))

        combined_diff = diff_local + diff_global

        # score fusion
        LGST_score = (combined_diff-self.lgst_mean)/self.lgst_std
        patch_hist_score = (patch_hist_score-self.patch_hist_mean)/self.patch_hist_std

        return [LGST_score,patch_hist_score]
    
def tflite_model_postprocess(model_output):
    LGST_score = np.max(model_output[0],axis=(1,2))
    patch_hist_score = model_output[1]
    return LGST_score + patch_hist_score