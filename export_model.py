import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import timm
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import os
import tqdm
import argparse

from benchmark_speed import FixedPatchClassDetector
from data_loader import MVTecLOCODataset

    
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
    def __init__(self,encoder,teacher,local_st,autoencoder,segmentor,patch_hist_detector,params):
        super(CSAD_ONNX, self).__init__()
        self.teacher_mean = params['teacher_mean']
        self.teacher_std = params['teacher_std']
        
        self.q_st_start = params['q_st_start']
        self.q_st_end = params['q_st_end']
        self.q_ae_start = params['q_ae_start']
        self.q_ae_end = params['q_ae_end']

        self.lgst_mean = params['lgst_mean']
        self.lgst_std = params['lgst_std']

        self.patch_hist_mean = params['patchhist_mean']
        self.patch_hist_std = params['patchhist_std']

        self.encoder = encoder
        self.teacher = teacher
        self.local_st = local_st
        self.autoencoder = autoencoder
        self.segmentor = segmentor
        self.patch_hist_detector = patch_hist_detector



    def forward(self, image):
        # shape: (B,3,256,256)
        feat = self.encoder(image)

        # Patch Histogram branch
        segmap = self.segmentor(feat)
        segmap = torch.argmax(segmap,dim=1)
        patch_hist_score = self.patch_hist_detector.detect_grid(segmap)

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

        LGST_score = torch.max(diff_local+diff_global)

        # score fusion
        patch_hist_score = (patch_hist_score-self.patch_hist_mean)/self.patch_hist_std
        LGST_score = (LGST_score-self.lgst_mean)/self.lgst_std

        return LGST_score+patch_hist_score
    

def convert_to_onnx(category):
    print("Loading checkpoint...")
    lgst_ckpt = torch.load(f"./ckpt/best_{category}.pth")
    teacher = ResNetTeacher(out_dim=512,feat_size=64)
    teacher.proj.weight = lgst_ckpt['teacher'].proj.weight
    encoder = timm.create_model('hf_hub:timm/wide_resnet50_2.tv2_in1k'
                                          ,pretrained=True,
                                          features_only=True,
                                          out_indices=[1,2,3])
    local_st = lgst_ckpt['student']
    autoencoder = lgst_ckpt['autoencoder']
    q_st_start = lgst_ckpt['q_st_start']
    q_st_end = lgst_ckpt['q_st_end']
    q_ae_start = lgst_ckpt['q_ae_start']
    q_ae_end = lgst_ckpt['q_ae_end']
    teacher_mean =  lgst_ckpt['teacher_mean']
    teacher_std = lgst_ckpt['teacher_std']

    # load patch hist branch
    segmentor = torch.load(f"./ckpt/segmentor_{category}_256.pth")
    patch_hist_detector = FixedPatchClassDetector(num_classes=segmentor.fc2.conv3.out_channels
                                             ,segmap_size=256)
    patch_hist_params = np.load(f"./anomaly_score/{category}_patchhist_params.npz")
    patch_hist_detector.hist_mean = torch.from_numpy(patch_hist_params['hist_mean']).cuda()
    patch_hist_detector.hist_invcov = torch.from_numpy(patch_hist_params['hist_invcov']).cuda()
    patch_hist_detector.patch_hist_mean = torch.from_numpy(patch_hist_params['patch_hist_mean']).cuda()
    patch_hist_detector.patch_hist_invcov = torch.from_numpy(patch_hist_params['patch_hist_invcov']).cuda()
    hist_val_score = patch_hist_params['hist_val_score']
    q = np.quantile(hist_val_score,0.2)
    p = np.quantile(hist_val_score,0.8)
    hist_val_score = hist_val_score[(hist_val_score>q) & (hist_val_score<p)]
    patch_hist_detector.hist_val_mean = torch.tensor(np.mean(hist_val_score),dtype=torch.float32).cuda()
    patch_hist_detector.hist_val_std = torch.tensor(np.std(hist_val_score),dtype=torch.float32).cuda()
    patch_hist_val_score = patch_hist_params['patch_hist_val_score']
    q = np.quantile(patch_hist_val_score,0.2)
    p = np.quantile(patch_hist_val_score,0.8)
    patch_hist_val_score = patch_hist_val_score[(patch_hist_val_score>q) & (patch_hist_val_score<p)]
    patch_hist_detector.patch_hist_val_mean = torch.tensor(np.mean(patch_hist_val_score),dtype=torch.float32).cuda()
    patch_hist_detector.patch_hist_val_std = torch.tensor(np.std(patch_hist_val_score),dtype=torch.float32).cuda()


    lgst_val = np.load(f"./anomaly_score/{category}_LGST_val_score.npy")
    q = np.quantile(lgst_val,0.2)
    p = np.quantile(lgst_val,0.8)
    lgst_val = lgst_val[(lgst_val>q) & (lgst_val<p)]
    lgst_mean = np.mean(lgst_val)
    lgst_std = np.std(lgst_val)

    patch_hist_val = np.load(f"./anomaly_score/{category}_patchhist_val_score.npy")
    q = np.quantile(patch_hist_val,0.2)
    p = np.quantile(patch_hist_val,0.8)
    patch_hist_val = patch_hist_val[(patch_hist_val>q) & (patch_hist_val<p)]
    patch_hist_mean = np.mean(patch_hist_val)
    patch_hist_std = np.std(patch_hist_val)

    params = {
        'teacher_mean':teacher_mean,
        'teacher_std':teacher_std,
        'q_st_start':q_st_start,
        'q_st_end':q_st_end,
        'q_ae_start':q_ae_start,
        'q_ae_end':q_ae_end,
        'lgst_mean':torch.tensor(lgst_mean,dtype=torch.float32).cuda(),
        'lgst_std':torch.tensor(lgst_std,dtype=torch.float32).cuda(),
        'patchhist_mean':torch.tensor(patch_hist_mean,dtype=torch.float32).cuda(),
        'patchhist_std':torch.tensor(patch_hist_std,dtype=torch.float32).cuda()
    }

    csad = CSAD_ONNX(
        encoder=encoder,
        teacher=teacher,
        local_st=local_st,
        autoencoder=autoencoder,
        segmentor=segmentor,
        patch_hist_detector=patch_hist_detector,
        params=params
    )
    print("Model loaded successfully!")

    csad.cuda().eval()

    print("Testing...")


    test_set = MVTecLOCODataset(
            root="./datasets/mvtec_loco_anomaly_detection",
            image_size=256,
            phase='test',
            category=category
        )
    test_set = DataLoader(test_set, batch_size=1, shuffle=False)

    logi_ture = []
    logi_score = []
    stru_ture = []
    stru_score = []
    for i,sample in tqdm.tqdm(enumerate(test_set), desc="Testing"):
        image = sample['image']
        path = sample['path']
        score = csad(image)
        score = score.item()
        defect_class = os.path.basename(os.path.dirname(path[0]))
        
        if defect_class == "good":
            logi_ture.append(0)
            logi_score.append(score)
            stru_ture.append(0)
            stru_score.append(score)
        elif defect_class == "logical_anomalies":
            logi_ture.append(1)
            logi_score.append(score)
        elif defect_class == "structural_anomalies":
            stru_ture.append(1)
            stru_score.append(score)

    logi_auc = roc_auc_score(y_true=logi_ture, y_score=logi_score)
    stru_auc = roc_auc_score(y_true=stru_ture, y_score=stru_score)
    print(f"Test result: logical auc:{logi_auc}, structural auc:{stru_auc}")

    
    torch_input = torch.randn(1,3,256,256).cuda()
    os.makedirs("./ckpt/onnx_models", exist_ok=True)

    # Export the model
    torch.onnx.export(csad,               # model being run
                  torch_input,                         # model input (or a tuple for multiple inputs)
                  f"./ckpt/onnx_models/{category}.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                #   do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}}
                )

    import onnx
    onnx_model = onnx.load(f"./ckpt/onnx_models/{category}.onnx")
    onnx.checker.check_model(onnx_model)

    print("Model saved successfully!")

    return (logi_auc*100+stru_auc*100)/2


def inference_onnx(category):
    # Inference after converting to onnx
    import onnxruntime
    if torch.cuda.is_available():
        providers = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'})]
    else:
        providers = ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(
        f"./ckpt/onnx_models/{category}.onnx", 
        providers=[("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'})])
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


    test_set = MVTecLOCODataset(
            root="./datasets/mvtec_loco_anomaly_detection",
            image_size=256,
            phase='test',
            category=category
        )
    test_set = DataLoader(test_set, batch_size=1, shuffle=False)

    logi_ture = []
    logi_score = []
    stru_ture = []
    stru_score = []
    for i,sample in tqdm.tqdm(enumerate(test_set), desc="Testing"):
        image = sample['image']
        path = sample['path']
        
        torch_input = image
        onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), [torch_input])}
        
        onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]
        score = onnxruntime_outputs[0]
        defect_class = os.path.basename(os.path.dirname(path[0]))
        
        if defect_class == "good":
            logi_ture.append(0)
            logi_score.append(score)
            stru_ture.append(0)
            stru_score.append(score)
        elif defect_class == "logical_anomalies":
            logi_ture.append(1)
            logi_score.append(score)
        elif defect_class == "structural_anomalies":
            stru_ture.append(1)
            stru_score.append(score)

    logi_auc = roc_auc_score(y_true=logi_ture, y_score=logi_score)
    stru_auc = roc_auc_score(y_true=stru_ture, y_score=stru_score)
    
    print(f"Test result: logical auc:{logi_auc*100}, structural auc:{stru_auc*100}")

    return (logi_auc*100+stru_auc*100)/2


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--only_inference",default=False, action="store_true")
    args = argparser.parse_args()
    categories = ["breakfast_box","juice_bottle","pushpins","screw_bag","splicing_connectors",]#
    aucs = []
    for category in categories:
        if args.only_inference:
            auc = inference_onnx(category)
        else:
            auc = convert_to_onnx(category)
        aucs.append(auc)
    print("Total Average AUC:",np.mean(aucs))