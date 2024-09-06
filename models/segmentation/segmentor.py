import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import tqdm
import timm
from models.segmentation.model import Segmentor
import os
import sklearn

from models.segmentation.loss import MulticlassCrossEntropyLoss, FocalLoss, MulticlassDiceLoss, ClassBalancedDiceLoss, HistLoss, EntropyLoss
from models.segmentation.patch_histogram import test_patch_histogram

def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)


        
    

class SegmentorTrainer():
    def __init__(self,encoder,config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pad2resize_linear = self.config['pad2resize']
        self.encoder = encoder
        self.encoder.to(self.device)
        self.encoder.eval()
        #self.encoder.eval()
        
        self.image_size = config['image_size']
        self.num_classes = np.load(self.config['mask_root']+"/info/filtered_component_histogram.npy").shape[1] + 1
        os.makedirs(self.config['mask_root']+"/test_seg_output",exist_ok=True)
        self.segmentor = Segmentor(
            in_dim=self.config['in_dim'],
            num_classes=self.num_classes,
            in_size=self.image_size,
            pad2resize=self.pad2resize_linear
        )
        self.segmentor.to(self.device)

        self.color_list = [[127, 123, 229], [195, 240, 251], [120, 200, 255],
               [243, 241, 230], [224, 190, 144], [178, 116, 75],
               [255, 100, 0], [0, 255, 100],
              [100, 0, 255], [100, 255, 0], [255, 0, 255],
              [0, 255, 255], [192, 192, 192], [128, 128, 128],
              [128, 0, 0], [128, 128, 0], [0, 128, 0],
              [128, 0, 128], [0, 128, 128], [0, 0, 128]]
        
    def fit(self,sup_dataloader,unsup_dataloader,val_dataloader,test_dataloader,full_train_dataloader):
        # training
        

        self.optimizer = torch.optim.Adam(self.segmentor.parameters(),lr=self.config["lr"],weight_decay=1e-5)
        self.ce_loss = MulticlassCrossEntropyLoss(ignore_index=None)
        self.focal_loss = FocalLoss(ignore_index=None)
        # self.dice_loss = MulticlassDiceLoss()
        self.dice_loss = ClassBalancedDiceLoss(ignore_index=None)
        self.hist_loss = HistLoss(ignore_index=0)
        self.entropy_loss = EntropyLoss()

        self.loss_dict = {
            "ce":self.ce_loss,
            "focal":self.focal_loss,
            "dice":self.dice_loss,
            "hist":self.hist_loss,
            "entropy":self.entropy_loss
        }
        self.loss_weight = self.config["loss_weight"]

        tqdm_obj = tqdm.tqdm(range(self.config["epoch"]),total=self.config["epoch"],desc="Training segmentor...")
        best_auc = 0
        best_logi_auc = 0
        best_strc_auc = 0

        iters_per_epoch = len(sup_dataloader)
        sup_dataloader_inf = InfiniteDataloader(sup_dataloader)
        unsup_dataloader_inf = InfiniteDataloader(unsup_dataloader)
        print(f"Start training {self.config['category']}...")
        for epoch in tqdm_obj:
            sup_only = True if epoch < self.config["sup_only_epoch"] else False
            self.segmentor.train()
            loss = self.train_one_epoch(tqdm_obj,sup_dataloader_inf,unsup_dataloader_inf,iters_per_epoch,sup_only)
            with torch.no_grad():
                if (epoch+1)%2 == 0:
                    print("Testing")
                    self.segmentor.eval()
                    self.test(test_dataloader)
                    
                    # logi_auc, struc_auc = self.test_hist_mahalanobis(self.segmentor,full_train_dataloader,val_dataloader,test_dataloader,num_classes=self.num_classes)
                    logi_auc, struc_auc = test_patch_histogram(
                        train_loader=full_train_dataloader,
                        val_loader=val_dataloader,
                        test_loader=test_dataloader,
                        encoder=self.encoder,
                        segmentor=self.segmentor,
                        category=self.config['category'],
                        patch_size=[256,128],
                        overlap_ratio=[0,0],
                        save_score=False
                    )
                    
                    print(f"logical AUC: {logi_auc:.4f}| structural AUC: {struc_auc:.4f}")
                    
                    
                    
                    if (logi_auc+struc_auc)/2 > best_auc:
                        print("Saving prediction, model and score...")
                        self.segmentor.eval()
                        best_strc_auc = struc_auc
                        best_logi_auc = logi_auc
                        best_auc = (logi_auc+struc_auc)/2
                        # save prediction of test image
                        self.test(test_dataloader)
                        # save prediction of all train image
                        self.test(full_train_dataloader,save_train_image=True)
                        # save prediction of train image(augmented)(supervised)
                        self.validate(sup_dataloader,sup=True)
                        # save prediction of train image(unsupervised)
                        self.validate(unsup_dataloader,sup=False)
                        # save model
                        self.save(self.config["model_path"])
                        # store anomaly score
                        logi_auc, struc_auc = test_patch_histogram(
                            train_loader=full_train_dataloader,
                            val_loader=val_dataloader,
                            test_loader=test_dataloader,
                            encoder=self.encoder,
                            segmentor=self.segmentor,
                            category=self.config['category'],
                            patch_size=[256,128],
                            overlap_ratio=[0,0],
                            save_score=True
                        )
        print(f"{self.config['category']} done!")
        print(f"Best AUC: {best_auc:.4f}| Best logical AUC: {best_logi_auc:.4f}| Best structural AUC: {best_strc_auc:.4f}")

    def de_normalize(self,tensor):
        # tensor: (B,C,H,W)
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
            mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
            tensor = tensor * std.unsqueeze(0).unsqueeze(2).unsqueeze(3) + mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            return tensor[0]
        else:
            mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
            tensor = tensor * std.unsqueeze(0).unsqueeze(2).unsqueeze(3) + mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            return tensor

    
    def train_one_epoch(self,tqdm_obj,sup_dataloader,unsup_dataloader,iters_per_epoch,sup_only=False):
        epoch_loss = 0
        for i,sup_batch,unsup_batch in zip(range(iters_per_epoch),sup_dataloader,unsup_dataloader):
            self.segmentor.train()

            image, gt, rand_gt, gt_path = sup_batch
            unsup_image, _, _, _ = unsup_batch

            with torch.no_grad():
                image = self.encoder(image)
                unsup_image = self.encoder(unsup_image)

            sup_out = self.segmentor(image)
            unsup_out = self.segmentor(unsup_image)
            
            sup_loss = 0
            sup_ce = self.loss_dict["ce"](sup_out,gt) * self.loss_weight['ce']
            sup_focal = self.loss_dict["focal"](sup_out,gt) * self.loss_weight['focal']
            sup_dice = self.loss_dict["dice"](sup_out,gt) * self.loss_weight['dice']
            sup_entro = self.loss_dict["entropy"](sup_out) * self.loss_weight['entropy']
            #sup_hist = self.loss_dict["hist"](sup_out,rand_gt) * self.loss_weight['hist']
            sup_loss = sup_ce + sup_focal + sup_dice #+ sup_hist

            unsup_loss = 0
            unsup_hist = self.loss_dict["hist"](unsup_out,rand_gt) * self.loss_weight['hist']
            unsup_entro = self.loss_dict["entropy"](unsup_out) * self.loss_weight['entropy']
            unsup_loss = unsup_hist + unsup_entro

            if sup_only:
                total_loss = sup_loss
            else:
                total_loss = sup_loss + unsup_loss + sup_entro
            epoch_loss += total_loss.item()

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            if i % 10 == 0:
                tqdm_obj.set_description(
                    f"Current loss: {total_loss.item():.4f}| CE:{sup_ce.item():.4f}| Dice:{sup_dice.item():.4f}| Hist:{unsup_hist.item():.4f}| Focal:{sup_focal.item():.4f}| Entropy:{sup_entro.item():.4f}({unsup_entro.item():.4f}) ")
        return epoch_loss

    def save(self,path):
        torch.save(self.segmentor,path)
    
    def load(self,path):
        self.segmentor = torch.load(path)
        self.segmentor.to(self.device)

    def validate(self,dataloader,sup=False):
        with torch.no_grad():
            for i,batch in enumerate(dataloader):
                image, gt, rand_gt, gt_path = batch
                with torch.no_grad():
                    image_feat = self.encoder(image)
                out = self.segmentor(image_feat)
                for j in range(len(image)):
                    out_softmax = F.softmax(out[j],dim=0)
                    de_image = self.de_normalize(image[j])
                    de_image = de_image.squeeze(0).permute(1,2,0).cpu().numpy()
                    de_image = (de_image*255).astype(np.uint8)
                    de_image = cv2.cvtColor(de_image,cv2.COLOR_RGB2BGR)
                    out_softmax = torch.argmax(out_softmax,dim=0)
                    out_softmax = out_softmax.cpu().numpy()
                    color_out = np.zeros((self.image_size,self.image_size,3))
                    color_gt = np.zeros((self.image_size,self.image_size,3))
                    for k in range(1,self.num_classes):
                        color_out[out_softmax==k,:] = self.color_list[k-1]
                        color_gt[gt[j,0].cpu().numpy()==k,:] = self.color_list[k-1]
                    color_out = color_out.astype(np.uint8)
                    color_gt = color_gt.astype(np.uint8)
                    result = np.hstack([de_image,color_out,color_gt])
                    save_path = '/'.join(gt_path[j].replace("\\","/").split('/')[:-2])+"/val_seg_output"
                    image_name = gt_path[j].replace("\\","/").split("/")[-2]
                    os.makedirs(save_path,exist_ok=True)
                    cv2.imwrite(f'{save_path}/{"sup" if sup else "unsup"}_{image_name}.png',result)
                    

    def test(self,dataloader,save_train_image=False):
        with torch.no_grad():
            for i,batch in enumerate(dataloader):
                image,image_path = batch
                with torch.no_grad():
                    image_feat = self.encoder(image)
                out = self.segmentor(image_feat)

                out_softmax = F.softmax(out[0],dim=0)
                de_image = self.de_normalize(image[0])
                de_image = de_image.squeeze(0).permute(1,2,0).cpu().numpy()
                de_image = (de_image*255).astype(np.uint8)
                de_image = cv2.cvtColor(de_image,cv2.COLOR_RGB2BGR)
                out_softmax = torch.argmax(out_softmax,dim=0)
                out_softmax = out_softmax.cpu().numpy()
                color_out = np.zeros((self.image_size,self.image_size,3))
                color_gt = np.zeros((self.image_size,self.image_size,3))
                for k in range(1,self.num_classes):
                    color_out[out_softmax==k,:] = self.color_list[k-1]
                color_out = color_out.astype(np.uint8)
                save_path = image_path[0].replace("\\","/").replace("mvtec_loco_anomaly_detection","masks")
                anomaly_type = save_path.split("/")[-2]
                image_name = save_path.split("/")[-1].split(".")[0]
                save_dir = "/".join(os.path.dirname(save_path).split("/")[:-2])+"/test_seg_output"
                os.makedirs(save_dir,exist_ok=True)
                # Image.fromarray(color_out).save(f'{gt_path[j].replace("filtered_cluster_map","pred_segmap_color")}.png')
                if save_train_image:
                    Image.fromarray(out_softmax.astype(np.uint8)).save(f'{"/".join(os.path.dirname(save_path).split("/")[:-2])}/{image_name}/pred_segmap.png')
                else:
                    cv2.imwrite(f'{save_dir}/{anomaly_type}_{image_name}.png',color_out)

    
    
    
    def test_hist_mahalanobis(self,segmentor,train_loader,val_loader,test_loader,num_classes=3,save_score=False):
        num_classes = num_classes - 1
        segmentor.eval()
        def histogram(label_map,num_classes):
            hist = np.zeros(num_classes)
            for i in range(1,num_classes+1): # not include background
                hist[i-1] = (label_map == i).sum()
            hist = hist / label_map.size
            return hist
        true_score_logi = []
        pred_score_logi = []
        true_score_strc = []
        pred_score_strc = []
        segmentor.eval()
        # get train histograms
        train_hists = []
        with torch.no_grad():
            for image,path in train_loader:
                with torch.no_grad():
                    image_feat = self.encoder(image)
                label_map = segmentor(image_feat)
                label_map = label_map.argmax(1)[0].cpu().numpy()
                train_hists.append(histogram(label_map,num_classes))
        train_hists = np.stack(train_hists,axis=0)
        mean = np.mean(train_hists,axis=0)
        from scipy.spatial.distance import mahalanobis
        def dist(x,data,mean):
            if data.shape[1] == 1:
                return np.linalg.norm(x-mean)
            else:
                cov = np.cov(data.T)
                return mahalanobis(x,mean,np.linalg.pinv(cov))
            
        val_scores = []
        for image,path in val_loader:
            with torch.no_grad():
                with torch.no_grad():
                    image_feat = self.encoder(image)
                label_map = segmentor(image_feat)
                label_map = label_map.argmax(1)[0].cpu().numpy()
                hist = histogram(label_map,num_classes)
            score = dist(hist,train_hists,mean)
            val_scores.append(score)
        val_scores = np.array(val_scores)
        

        # get test histograms
        for image,path in test_loader:
            with torch.no_grad():
                with torch.no_grad():
                    image_feat = self.encoder(image)
                label_map = segmentor(image_feat)
                label_map = label_map.argmax(1)[0].cpu().numpy()
                hist = histogram(label_map,num_classes)
            score = dist(hist,train_hists,mean)
            
            

            save_path = path[0].replace("\\","/").replace("mvtec_loco_anomaly_detection","masks")
            anomaly_type = save_path.split("/")[-2]
            image_name = save_path.split("/")[-1].split(".")[0]
            if anomaly_type == "logical_anomalies":
                true_score_logi.append(1)
                pred_score_logi.append(score.item())
            elif anomaly_type == "structural_anomalies":
                true_score_strc.append(1)
                pred_score_strc.append(score.item())
            elif anomaly_type == "good":
                true_score_logi.append(0)
                true_score_strc.append(0)
                pred_score_logi.append(score.item())
                pred_score_strc.append(score.item())

        true_score_logi = np.array(true_score_logi)
        pred_score_logi = np.array(pred_score_logi)
        true_score_strc = np.array(true_score_strc)
        pred_score_strc = np.array(pred_score_strc)
        auc_logi = sklearn.metrics.roc_auc_score(true_score_logi,pred_score_logi)
        auc_strc = sklearn.metrics.roc_auc_score(true_score_strc,pred_score_strc)
        

        if save_score:
            category = save_path.split("/")[-4]
            np.save(f"./anomaly_score/{category}_hist_val_score.npy",val_scores)
            np.save(f"./anomaly_score/{category}_hist_logi_score.npy",pred_score_logi)
            np.save(f"./anomaly_score/{category}_hist_struc_score.npy",pred_score_strc)

        
        return auc_logi, auc_strc
    
if __name__ == "__main__":
    category = 'juice_bottle'
    config = {
        "image_path":f"C:/Users/kev30/Desktop/anomaly/EfficientAD-res/datasets/mvtec_loco_anomaly_detection/{category}/train/good/*.png",
        "mask_root":f"C:/Users/kev30/Desktop/anomaly/EfficientAD-res/datasets/masks/{category}",
        "model_path":f"C:/Users/kev30/Desktop/anomaly/EfficientAD-res/ckpt/segmentor_{category}.pth ",
        "in_dim":[256,1024],
        "load":False,
        "image_size":512,
        "lr":1e-3,
        "epoch":150,
        "loss_weight":{
            "ce":1,
            "dice":1,
            "hist_entropy":1
        }
    }

    # dataset = SegmentDataset(image_path=config['image_path'],
    #                                   mask_root=config['mask_root'])
    # dataloader = DataLoader(dataset,batch_size=4,shuffle=False)
    encoder = timm.create_model('wide_resnet50_2.tv2_in1k'
                                          ,pretrained=True,
                                          features_only=True,
                                          out_indices=[1,2,3])
    # encoder = timm.create_model('resnet18.tv_in1k'
    #                                       ,pretrained=True,
    #                                       features_only=True,
    #                                       out_indices=[1,2,3])
    # segmentor = Segmentor(encoder,in_dim=[64,256],num_classes=3).cuda()
    segmentor = Segmentor(encoder,in_dim=[256,1024],num_classes=3).cuda()
    segmentor.eval()
    a = torch.randn(1,3,256,256).cuda()
    out = segmentor(a)
    print(out.shape)




    # segmentor_trainer = SegmentorTrainer(encoder,config)
    # segmentor_trainer.eval()

    
    import time
    with torch.no_grad():
        times = []
        for i in range(2000):
            image = torch.randn(2,3,256,256,dtype=torch.float32).cuda()
            start = time.time()
            out = segmentor.predict(image)
            times.append(time.time()-start)
    print(np.mean(times[-100:]))

    print("Done!")