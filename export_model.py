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
from data_loader import MVTecLOCODataset

def load_torch_model(category):
    from models.onnx_model import ResNetTeacher,CSAD_ONNX,FixedPatchClassDetector
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
    hist_val_std = np.std(hist_val_score) if np.std(hist_val_score) > 1e-3 else 1
    detector_params['hist_val_std'] = torch.tensor(hist_val_std,dtype=torch.float32)
    
    patch_hist_val_score = patch_hist_params['patch_hist_val_score']
    q = np.quantile(patch_hist_val_score,0.2)
    p = np.quantile(patch_hist_val_score,0.8)
    patch_hist_val_score = patch_hist_val_score[(patch_hist_val_score>q) & (patch_hist_val_score<p)]
    detector_params['patch_hist_val_mean'] = torch.tensor(np.mean(patch_hist_val_score),dtype=torch.float32)
    patch_hist_val_std = np.std(patch_hist_val_score) if np.std(patch_hist_val_score) > 1e-3 else 1
    detector_params['patch_hist_val_std'] = torch.tensor(patch_hist_val_std,dtype=torch.float32)
    
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
        'encoder':encoder.eval(),
        'teacher':teacher.eval(),
        'local_st':local_st.eval(),
        'autoencoder':autoencoder.eval(),
        'segmentor':segmentor.eval(),
        'patch_hist_detector':patch_hist_detector.eval(),
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
    from models.onnx_model import CSAD_ONNX
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
    from models.onnx_model import CSAD_ONNX
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

def tflite_model_postprocess(model_output):
    LGST_score = np.max(model_output[0],axis=(1,2))
    patch_hist_score = model_output[1]
    return LGST_score + patch_hist_score

def torch2tflite(category):
    import ai_edge_torch
    from models.tflite_model import CSAD_TFLITE
    os.environ["PJRT_DEVICE"] = "CPU"

    model_path = f"./ckpt/pytorch_models/{category}.pth"
    csad_state_dict = torch.load(model_path)
    num_classes = csad_state_dict['segmentor.fc2.conv3.weight'].shape[0]
    csad = CSAD_TFLITE(dim=512,num_classes=num_classes)
    csad.load_state_dict(csad_state_dict)
    csad = csad.eval()

    print(f"Load {model_path} successfully!")
    sample_inputs = (torch.randn(1, 3, 256, 256),)


    torch_output = csad(*sample_inputs)
    LGST_score = torch.amax(torch_output[0],dim=(1,2))
    patch_hist_score = torch_output[1]
    torch_output = LGST_score + patch_hist_score

    edge_model = ai_edge_torch.convert(csad.eval(), sample_inputs)
    edge_output = edge_model(*sample_inputs)
    edge_output = tflite_model_postprocess(edge_output)
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

def inference_tflite(category):
    import ai_edge_torch
    from models.tflite_model import tflite_model_postprocess
    root = os.path.dirname(os.path.abspath(__file__))
    csad = ai_edge_torch.load(f"{root}/ckpt/tflite_models/{category}.tflite")
    print(f"Load {root}/ckpt/tflite_models/{category}.tflite successfully!")

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
        
        tflite_input = (image,)
        tflite_output = csad(*tflite_input)
        score = tflite_model_postprocess(tflite_output)
        
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

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    """
        Export modules to Pytorch model:
            python export_model.py --format torch

        Convert Pytorch model to ONNX model:
            python export_model.py --format onnx

        Convert ONNX model to OpenVINO model:
            python export_model.py --format openvino

        Convert Pytorch model to TFLite model:
            python export_model.py --format tflite

        Inference Pytorch model:
            python export_model.py --inference_only --format torch

        Inference ONNX model:
            python export_model.py --inference_only --format onnx
        
        Inference OpenVINO model:
            python export_model.py --inference_only --format openvino

        Inference TFLite model:
            python export_model.py --inference_only --format tflite
    """
    argparser.add_argument("--inference_only",default=False, action="store_true")
    argparser.add_argument("--format",default="torch",choices=["onnx","openvino","torch","tflite"])
    args = argparser.parse_args()
    categories = ["pushpins","breakfast_box","juice_bottle","screw_bag","splicing_connectors",]#
    aucs = []
    for category in categories:
        if args.inference_only:
            if args.format == "onnx":
                auc = inference_onnx(category,)
            elif args.format == "openvino":
                auc = inference_openvino(category)
            elif args.format == "torch":
                auc = inference_torch(category)
            elif args.format == "tflite":
                auc = inference_tflite(category)
            
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