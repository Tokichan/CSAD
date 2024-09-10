import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import timm
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from models.model import LocalStudent, AutoEncoder
from models.segmentation.model import Segmentor
import os
import tqdm
import argparse

from data_loader import MVTecLOCODataset

class FixedPatchClassDetector(nn.Module):
    """
        Optimized version of patch histogram detector for fast speed
        and support batched input.
        (Predefined patch sizes and overlap ratios)
        (not using EMPatches to extract patches)
    """
    def __init__(self,num_classes=5,segmap_size=256,use_nahalanobis=False):
        super(FixedPatchClassDetector, self).__init__()
        self.num_classes = num_classes
        self.segmap_size = segmap_size
        
        self.use_nahalanobis = use_nahalanobis if (self.num_classes-1) > 1 else False

        self.hist_mean = torch.nn.Parameter(data=torch.randn((self.num_classes)),requires_grad=False)
        self.hist_invcov = torch.nn.Parameter(data=torch.randn((self.num_classes,self.num_classes)),requires_grad=False)
        self.patch_hist_mean = torch.nn.Parameter(data=torch.randn((self.num_classes*4)),requires_grad=False)
        self.patch_hist_invcov = torch.nn.Parameter(data=torch.randn((self.num_classes*4,self.num_classes*4)),requires_grad=False)
        self.hist_val_mean = torch.nn.Parameter(data=torch.randn(1)[0],requires_grad=False)
        self.hist_val_std = torch.nn.Parameter(data=torch.randn(1)[0],requires_grad=False)
        self.patch_hist_val_mean = torch.nn.Parameter(data=torch.randn(1)[0],requires_grad=False)
        self.patch_hist_val_std = torch.nn.Parameter(data=torch.randn(1)[0],requires_grad=False)
        

    def set_params(self,params):
        self.hist_mean = torch.nn.Parameter(data=params['hist_mean'],requires_grad=False)
        self.hist_invcov = torch.nn.Parameter(data=params['hist_invcov'],requires_grad=False)
        self.patch_hist_mean = torch.nn.Parameter(data=params['patch_hist_mean'],requires_grad=False)
        self.patch_hist_invcov = torch.nn.Parameter(data=params['patch_hist_invcov'],requires_grad=False)
        self.hist_val_mean = torch.nn.Parameter(data=params['hist_val_mean'],requires_grad=False)
        self.hist_val_std = torch.nn.Parameter(data=params['hist_val_std'],requires_grad=False)
        self.patch_hist_val_mean = torch.nn.Parameter(data=params['patch_hist_val_mean'],requires_grad=False)
        self.patch_hist_val_std = torch.nn.Parameter(data=params['patch_hist_val_std'],requires_grad=False)


    @staticmethod
    def histogram(seg_map):
        _, max_indices = torch.max(seg_map, dim=1)
        # Create a tensor of zeros with the same shape as the input tensor
        out = torch.zeros_like(seg_map)
        # Set the max values to 1 by using advanced indexing
        out.scatter_(1, max_indices.unsqueeze(1), 1)
        hist = torch.mean(out,dim=[2,3])
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
                diff_hist = torch.matmul(torch.matmul(hist[i]-self.hist_mean,self.hist_invcov),hist[i]-self.hist_mean)**0.5
                diff_patchhist = torch.matmul(torch.matmul(patch_hist[i]-self.patch_hist_mean,self.patch_hist_invcov),patch_hist[i]-self.patch_hist_mean)**0.5
            else:
                diff_hist = ((hist[i]-self.hist_mean)**2).sum()**0.5
                diff_patchhist = ((patch_hist[i]-self.patch_hist_mean)**2).sum()**0.5
            diff_hists.append(diff_hist)
            diff_patchhists.append(diff_patchhist)
        diff_hist = torch.stack(diff_hists)
        diff_patchhist = torch.stack(diff_patchhists)
        if self.hist_val_std > 1e-3:
            diff_hist = (diff_hist-self.hist_val_mean)/self.hist_val_std
        else:
            diff_hist = diff_hist-self.hist_val_mean
        if self.patch_hist_val_std > 1e-3:
            diff_patchhist = (diff_patchhist-self.patch_hist_val_mean)/self.patch_hist_val_std
        else:
            diff_patchhist = diff_patchhist-self.patch_hist_val_mean
        return diff_hist+diff_patchhist
    
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

        LGST_score = torch.max(diff_local+diff_global)

        # score fusion
        LGST_score = (LGST_score-self.lgst_mean)/self.lgst_std
        patch_hist_score = (patch_hist_score-self.patch_hist_mean)/self.patch_hist_std



        return LGST_score+patch_hist_score
    



def load_torch_model(category):
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
    patch_hist_detector = FixedPatchClassDetector(
        num_classes=segmentor.fc2.conv3.out_channels
        ,segmap_size=256,
        use_nahalanobis=True
    )
    patch_hist_params = np.load(f"./anomaly_score/{category}_patchhist_params.npz")
    detector_params = {}
    detector_params['hist_mean'] = torch.tensor(patch_hist_params['hist_mean'],dtype=torch.float32)
    detector_params['hist_invcov'] = torch.tensor(patch_hist_params['hist_invcov'],dtype=torch.float32)
    detector_params['patch_hist_mean'] = torch.tensor(patch_hist_params['patch_hist_mean'],dtype=torch.float32)
    detector_params['patch_hist_invcov'] = torch.tensor(patch_hist_params['patch_hist_invcov'],dtype=torch.float32)
    hist_val_score = patch_hist_params['hist_val_score']
    q = np.quantile(hist_val_score,0.2)
    p = np.quantile(hist_val_score,0.8)
    hist_val_score = hist_val_score[(hist_val_score>q) & (hist_val_score<p)]
    detector_params['hist_val_mean'] = torch.tensor(np.mean(hist_val_score),dtype=torch.float32)
    detector_params['hist_val_std'] = torch.tensor(np.std(hist_val_score),dtype=torch.float32)
    patch_hist_val_score = patch_hist_params['patch_hist_val_score']
    q = np.quantile(patch_hist_val_score,0.2)
    p = np.quantile(patch_hist_val_score,0.8)
    patch_hist_val_score = patch_hist_val_score[(patch_hist_val_score>q) & (patch_hist_val_score<p)]
    detector_params['patch_hist_val_mean'] = torch.tensor(np.mean(patch_hist_val_score),dtype=torch.float32)
    detector_params['patch_hist_val_std'] = torch.tensor(np.std(patch_hist_val_score),dtype=torch.float32)
    
    patch_hist_detector.set_params(detector_params)



    lgst_val = np.load(f"./anomaly_score/{category}_LGST_val_score.npy")
    q = np.quantile(lgst_val,0.2)
    p = np.quantile(lgst_val,0.8)
    lgst_val = lgst_val[(lgst_val>q) & (lgst_val<p)]
    lgst_mean = np.mean(lgst_val)
    lgst_std = np.std(lgst_val) if np.std(lgst_val) > 1e-3 else 1

    patch_hist_val = np.load(f"./anomaly_score/{category}_patchhist_val_score.npy")
    q = np.quantile(patch_hist_val,0.2)
    p = np.quantile(patch_hist_val,0.8)
    patch_hist_val = patch_hist_val[(patch_hist_val>q) & (patch_hist_val<p)]
    patch_hist_mean = np.mean(patch_hist_val)
    patch_hist_std = np.std(patch_hist_val) if np.std(patch_hist_val) > 1e-3 else 1

    params = {
        'teacher_mean':teacher_mean,
        'teacher_std':teacher_std,
        'q_st_start':q_st_start,
        'q_st_end':q_st_end,
        'q_ae_start':q_ae_start,
        'q_ae_end':q_ae_end,
        'lgst_mean':torch.tensor(lgst_mean,dtype=torch.float32),
        'lgst_std':torch.tensor(lgst_std,dtype=torch.float32),
        'patchhist_mean':torch.tensor(patch_hist_mean,dtype=torch.float32),
        'patchhist_std':torch.tensor(patch_hist_std,dtype=torch.float32)
    }

    models = {
        'encoder':encoder,
        'teacher':teacher,
        'local_st':local_st,
        'autoencoder':autoencoder,
        'segmentor':segmentor,
        'patch_hist_detector':patch_hist_detector,
        'params':params
    }

    csad = CSAD_ONNX.from_pretrained(models,params)
    return csad


def torch_export_model(category):
    csad = load_torch_model(category)
    csad.eval().cuda()

    os.makedirs("./ckpt/pytorch_models", exist_ok=True)
    torch.save(csad.state_dict(),f"./ckpt/pytorch_models/{category}.pth")
    print(f"Model saved to ./ckpt/pytorch_models/{category}.pth")

def torch2onnx(category):
    import onnx
    model_path = f"./ckpt/pytorch_models/{category}.pth"
    csad_state_dict = torch.load(model_path)
    num_classes = csad_state_dict['segmentor.fc2.conv3.weight'].shape[0]
    csad = CSAD_ONNX(dim=512,num_classes=num_classes)
    csad.load_state_dict(csad_state_dict)
    csad.cuda().eval()
    print("Model loaded successfully!")

    torch_input = torch.randn(1,3,256,256).cuda()
    os.makedirs("./ckpt/onnx_models", exist_ok=True)

    # Export the model
    torch.onnx.export(csad,               # model being run
        torch_input,                         # model input (or a tuple for multiple inputs)
        f"./ckpt/onnx_models/{category}.onnx",   # where to save the model (can be a file or file-like object)
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=17,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                    'output' : {0 : 'batch_size'}}
    )

    
    onnx_model = onnx.load(f"./ckpt/onnx_models/{category}.onnx")
    onnx.checker.check_model(onnx_model)
    print("Model successfully convert to ONNX!")
    

def inference_torch(category):
    model_path = f"./ckpt/pytorch_models/{category}.pth"
    csad_state_dict = torch.load(model_path)
    num_classes = csad_state_dict['segmentor.fc2.conv3.weight'].shape[0]
    csad = CSAD_ONNX(dim=512,num_classes=num_classes)
    csad.load_state_dict(csad_state_dict)
    csad.cuda().eval()
    print(f"Load {model_path} successfully!")
    
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
        out = csad(torch_input)
        score = out.item()
        # print(score)
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


def inference_onnx(category):
    # Inference after converting to onnx
    import onnxruntime
    if torch.cuda.is_available():
        use_cuda = True
        providers = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'})]
    else:
        use_cuda = False
        providers = ["CPUExecutionProvider"]


    model_path = f"./ckpt/onnx_models/{category}.onnx"

    ort_session = onnxruntime.InferenceSession(
        model_path, 
        providers=providers)
    print(f"Load {model_path} successfully!")
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


    test_set = MVTecLOCODataset(
            root="./datasets/mvtec_loco_anomaly_detection",
            image_size=256,
            phase='test',
            category=category,
            to_gpu=use_cuda
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
        # print(score)
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

def onnx2openvino(category):
    import shutil
    root = os.path.dirname(os.path.abspath(__file__))
    model_path = f"{root}/ckpt/onnx_models/{category}.onnx"

    if not os.path.exists(model_path):
        torch2onnx(category)

    os.system(f"mo --input_model {model_path} --compress_to_fp16=False")
    os.makedirs(f"{root}/ckpt/openvino_models",exist_ok=True)
    shutil.move(f"{root}/{category}.xml",f"{root}/ckpt/openvino_models/{category}.xml")
    shutil.move(f"{root}/{category}.bin",f"{root}/ckpt/openvino_models/{category}.bin")

def inference_openvino(category):
    # Inference after converting to onnx
    import openvino as ov
    core =  ov.Core()
    root = os.path.dirname(os.path.abspath(__file__))
    compiled_model = core.compile_model(f"{root}/ckpt/openvino_models/{category}.xml", "AUTO")
    infer_request = compiled_model.create_infer_request()
    
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


    test_set = MVTecLOCODataset(
            root="./datasets/mvtec_loco_anomaly_detection",
            image_size=256,
            phase='test',
            category=category,
            to_gpu=False
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


        # Create tensor from external memory
        input_tensor = ov.Tensor(array=to_numpy(torch_input), shared_memory=True)
        # Set input tensor for model with one input
        infer_request.set_input_tensor(input_tensor)

        infer_request.start_async()
        infer_request.wait()

        # Get output tensor for model with one output
        output = infer_request.get_output_tensor()
        output_buffer = output.data
        score = output_buffer[0]
        # print(score)


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


def torch2tflite(category):
    import ai_edge_torch
    model_path = f"./ckpt/pytorch_models/{category}.pth"
    csad_state_dict = torch.load(model_path)
    num_classes = csad_state_dict['segmentor.fc2.conv3.weight'].shape[0]
    csad = CSAD_ONNX(dim=512,num_classes=num_classes)
    csad.load_state_dict(csad_state_dict)
    csad.cuda().eval()
    print(f"Load {model_path} successfully!")
    sample_input = (torch.randn(1,3,256,256).cuda())
    torch_output = csad(*sample_input)
    edge_model = ai_edge_torch.convert(csad, sample_input)
    edge_output = edge_model(*sample_input)
    if (np.allclose(
        torch_output.detach().numpy(),
        edge_output,
        atol=1e-5,
        rtol=1e-5,
    )):
        print("Inference result with Pytorch and TfLite was within tolerance")
    else:
        print("Something wrong with Pytorch --> TfLite")

    root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(f"{root}/ckpt/tflite_models",exist_ok=True)
    edge_model.export(f"{root}/ckpt/tflite_models/{category}.tflite")
    print(f"Model saved to ./ckpt/tflite_models/{category}.tflite")



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    """
        Export modules to Pytorch model:
            python export_model.py --format torch

        Convert Pytorch model to ONNX model:
            python export_model.py --format onnx

        Convert ONNX model to OpenVINO model:
            python export_model.py --format openvino

        Convert Pytorch model to TFlite model:
            python export_model.py --format tflite

        Inference Pytorch model:
            python export_model.py --inference_only --format torch

        Inference ONNX model:
            python export_model.py --inference_only --format onnx
        
        Inference OpenVINO model:
            python export_model.py --inference_only --format openvino
    """
    argparser.add_argument("--inference_only",default=False, action="store_true")
    argparser.add_argument("--format",default="onnx",choices=["onnx","openvino","torch","tflite"])
    args = argparser.parse_args()
    categories = ["breakfast_box","juice_bottle","pushpins","screw_bag","splicing_connectors",]#
    aucs = []
    for category in categories:
        if args.inference_only:
            if args.format == "onnx":
                auc = inference_onnx(category,)
            elif args.format == "openvino":
                auc = inference_openvino(category)
            elif args.format == "torch":
                auc = inference_torch(category)
            
            aucs.append(auc)
        else:
            if args.format == "onnx":
                torch2onnx(category)
            elif args.format == "openvino":
                onnx2openvino(category)
            elif args.format == "torch":
                torch_export_model(category)
            elif args.format == "tflite":
                torch2tflite(category)

    if args.inference_only:
        print("Total Average AUC:",np.mean(aucs))