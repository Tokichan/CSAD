
import numpy as np
from empatches import EMPatches
import torch
import tqdm
import torch
import numpy as np
from empatches import EMPatches
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import mahalanobis

class PatchClassDetector():
    def __init__(self,num_classes,patch_size=16,overlap_ratio=0.5,segmap_size=256):
        self.num_classes = num_classes
        self.overlap_ratio = overlap_ratio
        self.segmap_size = segmap_size
        self.patch_size = patch_size 
        self.emp = EMPatches()
        self.hist_covs = None
        self.hist_means = None
        self.hists = []

    def aggregate_grid(self, segmap):
        # segmap: 1,H,W
        hist = self.image2hist(segmap).flatten()
        self.hists.append(hist)

    def image2hist(self,segmap):
        segmap = segmap[0,:,:,None]
        img_patches, indices = self.emp.extract_patches(segmap, patchsize=self.patch_size, overlap=self.overlap_ratio)
        #
        # reshape
        side_num = int(np.sqrt(len(img_patches)))
        # side_num*side_num, patch_size, patch_size, 1
        img_patches = np.expand_dims(np.array(img_patches),axis=1)[:,:,:,:,0]
        # img_patches = np.array(img_patches).reshape(side_num, side_num,1,self.patch_size,self.patch_size)
        # img_patches = img_patches.transpose(1,0,2,3,4)
        # hist = np.array([histogram(img_patches[i][j],self.num_classes) for i in range(side_num) for j in range(side_num)])
        hist = np.array([histogram(img_patches[i],self.num_classes) for i in range(side_num*side_num)])
        return hist

    def detect_grid(self, segmap):
        if self.hist_means is None:
            self.hist_means = np.mean(np.array(self.hists),axis=0)
            self.hist_covs = np.linalg.pinv(np.cov(np.array(self.hists).T))
        # 
        hist = self.image2hist(segmap).flatten()
        diff_hist = mahalanobis(hist,self.hist_means,self.hist_covs)
        if np.isnan(diff_hist):
            diff_hist = np.sqrt(np.sum((hist-self.hist_means)**2))
        return diff_hist

#915 126 854 201
# patchhist[0.001220703125, 0.0, 0.26934814453125, 0.0, 0.10260009765625, 0.53717041015625, 0.0, 0.0067138671875, 0.22418212890625, 0.09930419921875, 0.0006103515625, 0.01409912109375, 0.500244140625, 0.0, 0.0028076171875, 0.0, 0.0, 0.0, 0.66278076171875, 0.24420166015625, 0.0, 0.1014404296875, 0.0, 0.0, 0.1434326171875, 0.0943603515625, 0.5098876953125, 0.0]
# mean[0.0023688943977029915, 0.1210483648838141, 0.19975902416087962, 0.0006350438479344729, 0.1550018223602208, 0.43419123375178065, 0.0, 0.006460858206463675, 0.1109481768051104, 0.19063122523815884, 0.0007783286591880342, 0.08806312321937322, 0.4499656046897258, 0.0, 0.0029528147814280627, 6.173071358618234e-05, 0.0, 0.00017806267806267807, 0.6202270855591168, 0.28729995771011396, 0.0, 0.1534491644965278, 2.6778957443019944e-05, 1.9127826745014246e-06, 0.06091239038016382, 0.1632914746928419, 0.47288908781828703, 0.0]
#  diff=4.119
#tensor([[3.0518e-04, 0.0000e+00, 6.7337e-02, 0.0000e+00, 2.5650e-02, 1.3429e-01,
        #  0.0000e+00, 1.6785e-03, 5.6046e-02, 2.4826e-02, 1.5259e-04, 3.5248e-03,
        #  1.2506e-01, 0.0000e+00, 7.0190e-04, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #  1.6570e-01, 6.1050e-02, 0.0000e+00, 2.5360e-02, 0.0000e+00, 0.0000e+00,
        #  3.5858e-02, 2.3590e-02, 1.2747e-01, 0.0000e+00]], device='cuda:0')
def histogram(label_map,num_classes):
    hist = np.zeros(num_classes)
    for i in range(1,num_classes): # not include background
        hist[i-1] = (label_map == i).sum()
    hist = hist / label_map.size
    return hist 
    
def de_normalize(tensor):
    # tensor: (B,C,H,W)
    mean = torch.tensor([0.485, 0.456, 0.406]).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(tensor.device)
    tensor = tensor * std.unsqueeze(0).unsqueeze(2).unsqueeze(3) + mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    return tensor

def normalize(score,val_score):
    # trimmed z-normalization
    q = np.quantile(val_score,0.2)
    p = np.quantile(val_score,0.8)
    val_score = val_score[(val_score>q) & (val_score<p)]

    score = score-np.mean(val_score)
    score = score/np.std(val_score)
    return score

def test_patch_histogram(train_loader,val_loader,test_loader,encoder,segmentor,category,patch_size,overlap_ratio,save_score=False):
    patch_detectors = [PatchClassDetector(
        num_classes=segmentor.fc2.conv3.out_channels,
        patch_size=patch_size,
        overlap_ratio=overlap_ratio,
        segmap_size=256
    ) for patch_size,overlap_ratio in zip(patch_size,overlap_ratio)]
    
    
    # calculate mean and covariance matrix by train_loader
    for i, sample in tqdm.tqdm(enumerate(train_loader),desc='aggregate grid'):
        with torch.no_grad():
            image = sample[0]
            image.cuda()
            image = encoder(image)
            segmap = segmentor(image)
            segmap = torch.argmax(segmap, dim=1).cpu().numpy()
            for patch_detector in patch_detectors:
                patch_detector.aggregate_grid(segmap)

    # calculate anomaly score by val_loader
    val_scores = []
    for i, sample in tqdm.tqdm(enumerate(val_loader)):
        with torch.no_grad():
            image = sample[0]
            image.cuda()
            image = encoder(image)
            segmap = segmentor(image)
            segmap = torch.argmax(segmap, dim=1).cpu().numpy()
            diffs = []
            for patch_detector in patch_detectors:
                diff = patch_detector.detect_grid(segmap)
                diffs.append(diff)
            val_scores.append(diffs)
    val_scores = np.array(val_scores)
    
    logi_true_score = []
    struc_true_score = []
    logi_score = []
    struc_score = []
    for i, sample in tqdm.tqdm(enumerate(test_loader)):
        with torch.no_grad():
            image_path = sample[1][0]

            image = sample[0]
            image.cuda()
            image = encoder(image)
            segmap = segmentor(image)
            segmap = torch.argmax(segmap, dim=1).cpu().numpy()
            diffs = []
            for patch_detector in patch_detectors:
                diff = patch_detector.detect_grid(segmap)
                diffs.append(diff)
            if 'good' in image_path:
                logi_true_score.append(0)
                struc_true_score.append(0)
                logi_score.append(diffs)
                struc_score.append(diffs)
            elif 'logical_anomalies' in image_path:
                logi_true_score.append(1)
                logi_score.append(diffs)
            elif 'structural_anomalies' in image_path:
                struc_true_score.append(1)
                struc_score.append(diffs)
    logi_score = np.array(logi_score)
    struc_score = np.array(struc_score)
    logi_true_score = np.array(logi_true_score)
    struc_true_score = np.array(struc_true_score)

    # normalize logi_score and struc_score
    for j in range(logi_score.shape[1]):
        logi_score[:,j] = normalize(logi_score[:,j],val_scores[:,j])
    logi_score = np.sum(logi_score,axis=1)

    for j in range(struc_score.shape[1]):
        struc_score[:,j] = normalize(struc_score[:,j],val_scores[:,j])
    struc_score = np.sum(struc_score,axis=1)

    

    
    logi_auc = roc_auc_score(logi_true_score, logi_score)*100
    struc_auc = roc_auc_score(struc_true_score, struc_score)*100
    if save_score:
        # for onnx params
        params = {
            'hist_mean': patch_detectors[0].hist_means,
            'hist_invcov': patch_detectors[0].hist_covs,
            'patch_hist_mean': patch_detectors[1].hist_means,
            'patch_hist_invcov': patch_detectors[1].hist_covs,
            'hist_val_score': val_scores[:,0],
            'patch_hist_val_score': val_scores[:,1],
        }
        np.savez(f"./anomaly_score/{category}_patchhist_params.npz",**params)

        # normalize val_scores
        for j in range(val_scores.shape[1]):
            val_scores[:,j] = normalize(val_scores[:,j],val_scores[:,j])
        val_scores = np.sum(val_scores,axis=1)
        np.save(f"./anomaly_score/{category}_patchhist_logi_score.npy",logi_score)
        np.save(f"./anomaly_score/{category}_patchhist_struc_score.npy",struc_score)
        np.save(f"./anomaly_score/{category}_patchhist_val_score.npy",val_scores)
    return logi_auc,struc_auc
