import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np




class MulticlassCrossEntropyLoss(nn.Module):
    def __init__(self,ignore_index=None):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probabilities = logits

        probabilities = nn.Softmax(dim=1)(logits)
        # end if
        targets_one_hot = torch.nn.functional.one_hot(targets.squeeze(1), num_classes=logits.shape[1])
        # print(targets_one_hot.shape)
        # Convert from NHWC to NCHW
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).type(torch.float)

        if self.ignore_index is not None:
            targets_one_hot = torch.concat([targets_one_hot[:,:self.ignore_index],targets_one_hot[:,self.ignore_index+1:]],dim=1)
            probabilities = torch.concat([probabilities[:,:self.ignore_index],probabilities[:,self.ignore_index+1:]],dim=1)
        
        return self.ce_loss(probabilities,targets_one_hot)
    
class MulticlassDiceLoss(nn.Module):
    """Reference: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss
    """
    def __init__(self,ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets, reduction='mean', smooth=1e-6):
        """The "reduction" argument is ignored. This method computes the dice
        loss for all classes and provides an overall weighted loss.
        """
        probabilities = logits

        probabilities = nn.Softmax(dim=1)(logits)
        # end if
        targets_one_hot = torch.nn.functional.one_hot(targets.squeeze(1), num_classes=logits.shape[1])
        # print(targets_one_hot.shape)
        # Convert from NHWC to NCHW
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).type(torch.float)

        if self.ignore_index is not None:
            targets_one_hot = torch.concat([targets_one_hot[:,:self.ignore_index],targets_one_hot[:,self.ignore_index+1:]],dim=1)
            probabilities = torch.concat([probabilities[:,:self.ignore_index],probabilities[:,self.ignore_index+1:]],dim=1)
        
        # Multiply one-hot encoded ground truth labels with the probabilities to get the
        # prredicted probability for the actual class.
        intersection = (targets_one_hot * probabilities).sum()
        
        mod_a = intersection.sum()
        mod_b = targets.numel()
        
        dice_coefficient = 2. * intersection / (mod_a + mod_b + smooth)
        dice_loss = -dice_coefficient.log()
        return dice_loss
    
class ClassBalancedDiceLoss(nn.Module):
    def __init__(self,ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, prediction, target):
        probabilities = torch.softmax(prediction,dim=1)
        targets_one_hot = torch.nn.functional.one_hot(target.squeeze(1), num_classes=prediction.shape[1])
        # Convert from NHWC to NCHW
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).type(torch.float)

        if self.ignore_index is not None:
            targets_one_hot = torch.concat([targets_one_hot[:,:self.ignore_index],targets_one_hot[:,self.ignore_index+1:]],dim=1)
            probabilities = torch.concat([probabilities[:,:self.ignore_index],probabilities[:,self.ignore_index+1:]],dim=1)


        class_weights = self._calculate_class_weights(targets_one_hot)
        dice_loss = self._dice_loss(probabilities, targets_one_hot)
        class_balanced_loss = class_weights * dice_loss
        return class_balanced_loss.mean()

    def _calculate_class_weights(self, target):
        """
        Calculates class weights based on their inverse frequency in the target.
        """
        weights = torch.zeros((target.shape[0],target.shape[1])).cuda()
        for c in range(target.shape[1]):
            weights[:,c] = 1 / (target[:,c].sum() + 1e-5)
        weights = weights / weights.sum(dim=1,keepdim=True)
        return weights.detach()

    def _dice_loss(self, prediction, target):
        """
        Calculates dice loss for each class and then averages across all classes.
        """
        intersection = 2 * (prediction * target).sum(dim=(2, 3))
        union = prediction.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1e-5
        dice = (intersection + 1e-5) / (union + 1e-5)
        return 1 - dice
    

class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, ignore_index=None, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average
        self.ignore_index = ignore_index

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        logit = torch.softmax(logit, dim=1)
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.ignore_index is not None:
            one_hot_key = torch.concat([one_hot_key[:,:self.ignore_index],one_hot_key[:,self.ignore_index+1:]],dim=1)
            logit = torch.concat([logit[:,:self.ignore_index],logit[:,self.ignore_index+1:]],dim=1)


        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss
    

class OldHistLoss(nn.Module):
    def __init__(self,ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits,targets):
        # ignore background
        probabilities = logits

        probabilities = nn.Softmax(dim=1)(probabilities)

        targets_one_hot = torch.nn.functional.one_hot(targets.squeeze(1), num_classes=logits.shape[1])

        # print(targets_one_hot.shape)
        # Convert from NHWC to NCHW
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).type(torch.float)
    
        if self.ignore_index is not None:
            targets_one_hot = torch.concat([targets_one_hot[:,:self.ignore_index],targets_one_hot[:,self.ignore_index+1:]],dim=1)
            probabilities = torch.concat([probabilities[:,:self.ignore_index],probabilities[:,self.ignore_index+1:]],dim=1)
    
        
        targets_hist = torch.mean(targets_one_hot,dim=(2,3)) # (B,C)
        targets_hist = torch.nn.functional.normalize(targets_hist,dim=1)
        
        preds_hist = torch.mean(probabilities,dim=(2,3)) # (B,C)
        preds_hist = torch.nn.functional.normalize(preds_hist,dim=1)

        hist_loss = torch.mean(torch.abs(targets_hist-preds_hist))
        return hist_loss
    
class HistLoss(nn.Module):
    def __init__(self,ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self,pred, trg):
        pred = torch.softmax(pred,dim=1)
        new_trg = torch.zeros_like(trg).repeat(1, pred.shape[1], 1, 1).long()
        new_trg = new_trg.scatter(1, trg, 1).float()
        diff = torch.abs(new_trg.mean((2, 3)) - pred.mean((2, 3)))
        if self.ignore_index is not None:
            diff = torch.concat([diff[:,:self.ignore_index],diff[:,self.ignore_index+1:]],dim=1)
        loss = diff.sum() / pred.shape[0]  # exclude BG
        return loss
    
class EntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,pred):
        prob = nn.Softmax(dim=1)(pred)
        return (-1*prob*((prob+1e-5).log())).mean()