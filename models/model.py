import torch.nn as nn
import torch
import torch.nn.functional as F
import timm
import math
class PDN_M(nn.Module):
    def __init__(self, out_dim=512,feat_size=64,padding=False):
        super(PDN_M, self).__init__()
        self.out_dim = out_dim
        self.st_size = feat_size
        self.ae_size = feat_size
        self.padding = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=3*self.padding)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3*self.padding)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1*self.padding)
        self.conv5 = nn.Conv2d(512, self.out_dim*2, kernel_size=4, stride=1, padding=0)
        self.conv6 = nn.Conv2d(self.out_dim*2, self.out_dim*2, kernel_size=1, stride=1, padding=0)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1*self.padding)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1*self.padding)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.avgpool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        # x = F.relu(x)
        x = F.interpolate(x, size=self.st_size, mode='bilinear', align_corners=False)
        st = x[:,:self.out_dim]
        ae = x[:,self.out_dim:]
        return st,ae

        
class LocalStudent(nn.Module):

    def __init__(self, out_dim=512,feat_size=64,padding=False):
        super(LocalStudent, self).__init__()
        self.out_dim = out_dim
        self.feat_size = feat_size
        self.padding = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=3*self.padding)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3*self.padding)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1*self.padding)
        self.conv5 = nn.Conv2d(512, self.out_dim, kernel_size=4, stride=1, padding=0)
        # self.conv6 = nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, stride=1, padding=0)

        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1*self.padding)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1*self.padding)
        
        self.conv_st = nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, stride=1, padding=0)
        self.conv_ae = nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.avgpool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = F.interpolate(x, size=self.feat_size, mode='bilinear', align_corners=False)
        st = self.conv_st(x)
        ae = self.conv_ae(x)
        return st,ae
        
    
        
class AutoEncoder(nn.Module):
    def __init__(self, out_size=64,out_dim=512,base_dim=64,input_size=256,padding=True):
        super(AutoEncoder, self).__init__()
        self.out_dim = out_dim
        self.base_dim = base_dim
        self.input_size = input_size
        self.out_size = out_size
        self.padding = padding

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
        x = self.dropout1(x)
        x = F.interpolate(x, size=8, mode='bilinear')
        x = F.relu(self.deconv2(x))
        x = self.dropout2(x)
        x = F.interpolate(x, size=15, mode='bilinear')
        x = F.relu(self.deconv3(x))
        x = self.dropout3(x)
        x = F.interpolate(x, size=32, mode='bilinear')
        x = F.relu(self.deconv4(x))
        x = self.dropout4(x)
        x = F.interpolate(x, size=63, mode='bilinear')
        x = F.relu(self.deconv5(x))
        x = self.dropout5(x)
        x = F.interpolate(x, size=127, mode='bilinear')
        x = F.relu(self.deconv6(x))
        x = self.dropout6(x)
        x = F.interpolate(x, size=self.out_size, mode='bilinear')
        x = F.relu(self.deconv7(x))
        x = self.deconv8(x)
        return x
    
class ResNetTeacher(nn.Module):
    def __init__(self,out_dim=512,feat_size=64):
        super(ResNetTeacher, self).__init__()
        # hf_hub:timm/wide_resnet50_2.tv2_in1k
        # hf_hub:timm/wide_resnet50_2.racm_in1k
        self.encoder = timm.create_model('hf_hub:timm/wide_resnet50_2.tv2_in1k'
                                          ,pretrained=True,
                                          features_only=True,
                                          out_indices=[2,3])
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.out_dim = out_dim
        self.feat_size = feat_size
        self.proj = nn.Conv2d(1024+512, self.out_dim, kernel_size=1, stride=1, padding=0)
        self.proj.requires_grad_(False)
        
    def forward(self, x):
        x = self.encoder(x)
        concat_feat = []
        for i in range(len(x)):
            feat = x[i]
            feat = F.interpolate(feat, size=self.feat_size, mode='bilinear',align_corners=False)
            concat_feat.append(feat)
        concat_feat = torch.cat(concat_feat,dim=1)
        concat_feat = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(concat_feat)
        proj_feat = self.proj(concat_feat)
        return proj_feat
    
    


if __name__ == '__main__':
    import torch
    import time
    from PIL import Image
    import torchsummary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AutoEncoder(
        out_size=64,
        out_dim=512,
        base_dim=64,
        input_size=256
    ).cuda()
    torchsummary.summary(model, (3, 256, 256))

