#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tifffile
import torch

from torch.utils.data import DataLoader
from torchvision import transforms

import itertools
import os
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from data_loader import MVTecLOCODataset,InfiniteDataloader,ImageNetDataset
import yaml
from models.model import  AutoEncoder, ResNetTeacher, LocalStudent, PDN_M
from tensorboardX import SummaryWriter 
import datetime
import shutil
import cv2
import copy


def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def predict_stae(image, teacher, student, autoencoder, out_channels, teacher_mean, teacher_std,
                 q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None,):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    st_student_output,ae_student_output = student(image)

    autoencoder_output = autoencoder(image)
    diff_st = (teacher_output - st_student_output)**2
    diff_ae = (autoencoder_output - ae_student_output)**2
    map_st = torch.mean((teacher_output - st_student_output)**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output - ae_student_output)**2,
                        dim=1, keepdim=True)
    
    if q_st_start is not None:
        if (q_st_end - q_st_start)>1e-6:
            map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
        else:
            print(f'warn: q_st_end:{q_st_end} - q_st_start:{q_st_start} < 1e-6')
    if q_ae_start is not None:
        if (q_ae_end - q_ae_start)>1e-6:
            map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
        else:
            print(f'warn: q_ae_end:{q_ae_end} - q_ae_start:{q_ae_start} < 1e-6')

    # map_combined = map_st + map_ae
    return map_st, map_ae, diff_st, diff_ae

class CSAD:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.description = ""
        # set parameters
        self.config['output_dir'] = self.config['output_dir'] + f"/{self.config['run_name']}"
        self.seed = self.config['seed']
        self.description += f'seed:{self.seed}\n\n'
        self.on_gpu = torch.cuda.is_available()
        self.description += f'on_gpu:{self.on_gpu}\n\n'
        self.out_channels = self.config['Model']['channel_size']
        self.description += f'out_channels:{self.out_channels}\n\n'
        self.image_size = self.config['Model']['input_size']
        self.description += f'image_size:{self.image_size}\n\n'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.padding = self.config['Model']['padding']
        self.q_st_start = None
        self.q_st_end = None
        self.q_ae_start = None
        self.q_ae_end = None
        self.combined_mst_ratio = self.config['combined_mst_ratio']
        self.combined_mae_ratio = self.config['combined_mae_ratio']

        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = f'./results/{self.config["run_name"]}/{self.config["category"]}-{current_datetime}'
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        shutil.copyfile('./configs/mvtecloco_train.yaml',f'{self.log_dir}/config.yaml')
        shutil.copyfile('./models/model.py',f'{self.log_dir}/models.py')
        shutil.copyfile('./CSAD.py',f'{self.log_dir}/CSAD.py')

        seed_everything(seed=self.seed)



        self.dataset_path = self.config['Datasets']['train']['root']
        self.description += f"dataset:{self.config['Datasets']['train']['type']}\n\n"

        self.pretrain_penalty = True
        self.description += f"pretrain_penalty:{self.pretrain_penalty}\n\n"
        # create output dir
        self.train_output_dir = os.path.join(self.config['output_dir'], 'trainings',
                                        self.config['Datasets']['train']['type'], self.config['category'])
        self.test_output_dir = os.path.join(self.config['output_dir'], 'anomaly_maps',
                                    self.config['Datasets']['train']['type'], self.config['category'], 'test')
        os.makedirs(self.train_output_dir, exist_ok=True)
        os.makedirs(self.test_output_dir, exist_ok=True)

        # load data
        self.full_train_set = MVTecLOCODataset(
            root=self.dataset_path,
            image_size=self.config['Model']['input_size'],
            phase='train', 
            category=self.config['category'],
        )
        self.description += f"full_train_set len:{len(self.full_train_set)}\n\n"


        self.test_set = MVTecLOCODataset(
            root=self.dataset_path,
            image_size=self.config['Model']['input_size'],
            phase='test',
            category=self.config['category']
        )
        self.test_set = DataLoader(self.test_set, batch_size=1, shuffle=False)

        self.train_set = self.full_train_set
        self.validation_set = MVTecLOCODataset(
                                    root=self.dataset_path,
                                    image_size=self.config['Model']['input_size'],
                                    phase='eval', 
                                    category=self.config['category'],)

        self.description += f"train_set len:{len(self.train_set)}\n\n"
        self.description += f"validation_set len:{len(self.validation_set)}\n\n"


        self.train_loader = DataLoader(self.train_set, batch_size=self.config['Model']['batch_size'], shuffle=True, num_workers=0)



        self.train_loader_infinite = InfiniteDataloader(self.train_loader)
        self.validation_loader = DataLoader(self.validation_set, batch_size=1)

        if self.pretrain_penalty:
            # load pretraining data for penalty
            penalty_transform = transforms.Compose([
                transforms.Resize((2 * self.image_size, 2 * self.image_size)),
                transforms.RandomGrayscale(0.3),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                                    0.225])
            ])
            self.penalty_set = ImageNetDataset(penalty_transform)
            self.penalty_loader = DataLoader(self.penalty_set, batch_size=1, shuffle=True)
            self.penalty_loader_infinite = InfiniteDataloader(self.penalty_loader)
            self.description += f"penalty_set len:{len(self.penalty_set)}\n\n"
            self.description += f"penalty_transform:{penalty_transform}\n\n"
        else:
            self.penalty_loader_infinite = itertools.repeat(None)
        


        # create models

        self.student = LocalStudent(
            out_dim=512,feat_size=64,padding=True
        )
        self.autoencoder = AutoEncoder(out_size=64,out_dim=512,base_dim=64)
        self.teacher = ResNetTeacher(
            out_dim=512,
            feat_size=64,
        )


        self.description += f"#######################\n\n"
        self.description += f"teacher:{self.teacher}\n\n"
        self.description += f"#######################\n\n"
        self.description += f"student:{self.student}\n\n"
        self.description += f"#######################\n\n"
        self.description += f"autoencoder:{self.autoencoder}\n\n"
        self.description += f"#######################\n\n"



        # teacher frozen
        self.teacher.eval()
        self.student.train()
        self.autoencoder.train()

        if self.on_gpu:
            self.teacher.cuda()
            self.student.cuda()
            self.autoencoder.cuda()
        self.teacher_mean, self.teacher_std = self.teacher_normalization(self.train_loader)
        
        self.optimizer = torch.optim.Adam(itertools.chain(self.student.parameters(),
                                                    self.autoencoder.parameters()),
                                    lr=2e-4, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=int(0.95 * self.config['Model']['iterations']), gamma=0.1)
        self.loss_weights = self.config['Model']['loss_weights']
        

        self.description += f"optimizer:{self.optimizer}\n\n"
        self.description += f"scheduler: StepLR(step_size={int(0.95 * self.config['Model']['iterations'])},gamma={0.1})\n\n"
        self.description += f"loss_weights:{self.loss_weights}\n\n"

        with open(f'{self.log_dir}/description.txt','a') as f:
            f.write(f'{self.description}')

    @torch.no_grad()
    def teacher_normalization(self,train_loader):
        mean_outputs = []
        for sample in tqdm(train_loader, desc='Computing mean of features'):
            train_image = sample['image']
            teacher_output = self.teacher(train_image)
            mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
            mean_outputs.append(mean_output)
        channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
        channel_mean = channel_mean[None, :, None, None]

        mean_distances = []
        for sample in tqdm(train_loader, desc='Computing std of features'):
            train_image = sample['image']
            teacher_output = self.teacher(train_image)
            distance = (teacher_output - channel_mean) ** 2
            mean_distance = torch.mean(distance, dim=[0, 2, 3])
            mean_distances.append(mean_distance)
        channel_var = torch.mean(torch.stack(mean_distances), dim=0)
        channel_var = channel_var[None, :, None, None]
        channel_std = torch.sqrt(channel_var)

        return channel_mean, channel_std

    def val(self,test_set,iteration,save_img=False):
        y_true = []
        y_score = []
        local_maps = []
        global_maps = []
        scores = []
        for i,sample in tqdm(enumerate(test_set), desc="Validation"):
            image = sample['image']
            path = sample['path']
            orig_width = image.shape[3]
            orig_height = image.shape[2]

            map_st, map_ae,_,_ = predict_stae(
                image=image,
                teacher=self.teacher,
                student=self.student,
                autoencoder=self.autoencoder,
                out_channels=self.out_channels,
                teacher_mean=self.teacher_mean,
                teacher_std=self.teacher_std,
                q_st_start=self.q_st_start,
                q_st_end=self.q_st_end,
                q_ae_start=self.q_ae_start,
                q_ae_end=self.q_ae_end,
            )
            if not self.padding:
                map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
                map_st = torch.nn.functional.pad(map_st, (4, 4, 4, 4))
                map_ae = torch.nn.functional.pad(map_ae, (4, 4, 4, 4))


            map_st = torch.nn.functional.interpolate(
                map_st, (orig_height, orig_width), mode='bilinear')
            map_ae = torch.nn.functional.interpolate(
                map_ae, (orig_height, orig_width), mode='bilinear')

            map_st = map_st[0, 0].cpu().numpy()
            map_ae = map_ae[0, 0].cpu().numpy()

            local_maps.append(map_st)
            global_maps.append(map_ae)

            map_combined = map_st + map_ae
            score = np.max(map_combined)
            scores.append(score)
        if save_img:
            category = self.test_output_dir.replace("\\","/").split("/")[-2]
            os.makedirs(f'./anomaly_score',exist_ok=True)
            np.save(f'./anomaly_score/{category}_{self.config["run_name"]}_val_score.npy', np.array(scores))
       
    def test(self,test_set,iteration,save_img=False):
        logi_ture = []
        logi_score = []
        stru_ture = []
        stru_score = []
        local_maps = []
        global_maps = []
        seg_maps = []
        for i,sample in tqdm(enumerate(test_set), desc="Testing"):
            image = sample['image']
            path = sample['path']
            orig_width = image.shape[3]
            orig_height = image.shape[2]

            map_st, map_ae,diff_st,diff_ae = predict_stae(
                image=image,
                teacher=self.teacher,
                student=self.student,
                autoencoder=self.autoencoder,
                out_channels=self.out_channels,
                teacher_mean=self.teacher_mean,
                teacher_std=self.teacher_std,
                q_st_start=self.q_st_start,
                q_st_end=self.q_st_end,
                q_ae_start=self.q_ae_start,
                q_ae_end=self.q_ae_end,
            )
            if not self.padding:
                map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
                map_st = torch.nn.functional.pad(map_st, (4, 4, 4, 4))
                map_ae = torch.nn.functional.pad(map_ae, (4, 4, 4, 4))
           


            map_st = torch.nn.functional.interpolate(
                map_st, (orig_height, orig_width), mode='bilinear')
            map_ae = torch.nn.functional.interpolate(
                map_ae, (orig_height, orig_width), mode='bilinear')
            map_combined = map_st + map_ae

            map_combined = map_combined[0, 0].cpu().numpy()
            map_st = map_st[0, 0].cpu().numpy()
            map_ae = map_ae[0, 0].cpu().numpy()

            local_maps.append(map_st)
            global_maps.append(map_ae)


            defect_class = os.path.basename(os.path.dirname(path[0]))
            if save_img:
                img_nm = os.path.split(path[0])[1].split('.')[0]
                if not os.path.exists(os.path.join(self.test_output_dir, defect_class)):
                    os.makedirs(os.path.join(self.test_output_dir, defect_class))
                file = os.path.join(self.test_output_dir, defect_class, img_nm + '.tiff')
                tifffile.imwrite(file, map_combined)

            if self.writer is not None:
                img_nm = os.path.split(path[0])[1].split('.')[0]
                map_st = np.clip(map_st,0,1)
                map_st = np.uint8(255 * map_st)
                map_st = cv2.applyColorMap(map_st, cv2.COLORMAP_JET)
                map_st = cv2.cvtColor(map_st, cv2.COLOR_BGR2RGB)
                map_st = np.transpose(map_st,(2,0,1))/255

                map_ae = np.clip(map_ae,0,1)
                map_ae = np.uint8(255 * map_ae)
                map_ae = cv2.applyColorMap(map_ae, cv2.COLORMAP_JET)
                map_ae = cv2.cvtColor(map_ae, cv2.COLOR_BGR2RGB)
                map_ae = np.transpose(map_ae,(2,0,1))/255

                
                self.writer.add_image(f'{defect_class}/{img_nm}/local', map_st, iteration)
                self.writer.add_image(f'{defect_class}/{img_nm}/global', map_ae, iteration)
                
            if defect_class == "good":
                logi_ture.append(0)
                logi_score.append(np.max(map_combined))
                stru_ture.append(0)
                stru_score.append(np.max(map_combined))
            elif defect_class == "logical_anomalies":
                logi_ture.append(1)
                logi_score.append(np.max(map_combined))
            elif defect_class == "structural_anomalies":
                stru_ture.append(1)
                stru_score.append(np.max(map_combined))

        logi_auc = roc_auc_score(y_true=logi_ture, y_score=logi_score)
        stru_auc = roc_auc_score(y_true=stru_ture, y_score=stru_score)
        if save_img:
            category = self.test_output_dir.replace("\\","/").split("/")[-2]
            os.makedirs(f'./anomaly_score',exist_ok=True)
            np.save(f'./anomaly_score/{category}_{self.config["run_name"]}_logi_score.npy', np.array(logi_score))
            np.save(f'./anomaly_score/{category}_{self.config["run_name"]}_struc_score.npy', np.array(stru_score))
            # np.save(os.path.join(self.test_output_dir, 'global_maps.npy'), global_maps)
        return logi_auc*100, stru_auc*100
            
    @torch.no_grad()
    def map_normalization(self,validation_loader, desc='Map normalization'):
        maps_st = []
        maps_ae = []
        for sample in tqdm(validation_loader, desc=desc):
            image = sample['image']
            map_st, map_ae,_,_ = predict_stae(
                image=image,
                teacher=self.teacher,
                student=self.student,
                autoencoder=self.autoencoder,
                out_channels=self.out_channels,
                teacher_mean=self.teacher_mean,
                teacher_std=self.teacher_std,
            )
            maps_st.append(map_st)
            maps_ae.append(map_ae)

        maps_st = torch.cat(maps_st)
        maps_ae = torch.cat(maps_ae)

        q_st_start = torch.quantile(maps_st, q=0.9)
        q_st_end = torch.quantile(maps_st, q=0.995)
        q_ae_start = torch.quantile(maps_ae, q=0.9)
        q_ae_end = torch.quantile(maps_ae, q=0.995)

        return q_st_start, q_st_end, q_ae_start, q_ae_end

    def train(self):
        tqdm_obj = tqdm(range(config['Model']['iterations']))
        best_auc = 0
        best_auc_logi = 0
        best_auc_struc = 0
        best_iter = 0
        loss_dict = {'loss_st':[],'loss_ae':[],'loss_stae':[]}

        for iteration, sample, image_penalty in zip(tqdm_obj, self.train_loader_infinite, self.penalty_loader_infinite):
            image = sample['image']
            idx = sample['idx']

            with torch.no_grad():
                teacher_output = self.teacher(image)
                teacher_output = (teacher_output - self.teacher_mean) / self.teacher_std
            student_output_st,student_output_ae = self.student(image)
            distance_st = torch.pow(teacher_output - student_output_st, 2)
            d_hard = torch.quantile(distance_st, q=0.999)
            loss_hard = torch.mean(distance_st[distance_st >= d_hard])

            if image_penalty is not None:
                student_output_penalty,_ = self.student(image_penalty)
                loss_penalty = torch.mean(student_output_penalty**2)
                loss_st = loss_hard + loss_penalty
            else:
                loss_st = loss_hard


            # ae_output = self.autoencoder(image)
            # enable augmentation for global student
            image_ae = sample['aug_image']
            with torch.no_grad():
                teacher_output_ae = self.teacher(image_ae)
                teacher_output_ae = (teacher_output_ae - self.teacher_mean) / self.teacher_std
            teacher_output = teacher_output_ae
            _,student_output_ae = self.student(image_ae)
            ae_output = self.autoencoder(image_ae)


            distance_ae = torch.pow(teacher_output - ae_output, 2)
            distance_stae = torch.pow(ae_output - student_output_ae, 2)
            loss_ae = torch.mean(distance_ae)
            loss_stae = torch.mean(distance_stae)

            loss_dict['loss_st'].append(loss_st.item())
            loss_dict['loss_ae'].append(loss_ae.item())
            loss_dict['loss_stae'].append(loss_stae.item())

            loss_total = loss_st*self.loss_weights['st'] + loss_ae*self.loss_weights['ae'] + loss_stae*self.loss_weights['stae']
           

            self.optimizer.zero_grad()
            loss_total.backward()
            
            self.optimizer.step()
            self.scheduler.step()

            

            if iteration % 10 == 0:
                tqdm_obj.set_description(
                    f"Current loss: {loss_total.item():.4f}|st:{np.mean(loss_dict['loss_st']):.4f}|ae:{np.mean(loss_dict['loss_ae']):.4f}|stae:{np.mean(loss_dict['loss_stae']):.4f}|")
                self.writer.add_scalar('loss_st', np.mean(loss_dict['loss_st']), iteration)
                self.writer.add_scalar('loss_ae', np.mean(loss_dict['loss_ae']), iteration)
                self.writer.add_scalar('loss_stae', np.mean(loss_dict['loss_stae']), iteration)
                self.writer.add_scalar('total_loss', 
                                np.mean(loss_dict['loss_st'])+\
                                np.mean(loss_dict['loss_ae'])+\
                                np.mean(loss_dict['loss_stae']),
                                    iteration)
                loss_dict = {'loss_st':[],'loss_ae':[],'loss_stae':[]}

            if (iteration+1) % self.config['eval_freq'] == 0:
                # run intermediate evaluation
                self.teacher.eval()
                self.student.eval()
                self.autoencoder.eval()
                self.q_st_start, self.q_st_end, self.q_ae_start, self.q_ae_end = self.map_normalization(
                    validation_loader=self.validation_loader
                )


                logi_auc,struc_auc = self.test(test_set=self.test_set,iteration=iteration)
                auc = (logi_auc+struc_auc)/2
                
                print(f'\nIntermediate image auc of {config["category"]}: logi:{logi_auc:.4f}, struc:{struc_auc:.4f}, mean:{auc:.4f}')
                self.writer.add_scalar('logical_auc', logi_auc, iteration)
                self.writer.add_scalar('structural_auc', struc_auc, iteration)
                self.writer.add_scalar('auc', auc, iteration)

                # teacher frozen
                self.teacher.eval()
                self.student.train()
                self.autoencoder.train()

                if auc > best_auc:
                    self.teacher.eval()
                    self.student.eval()
                    self.autoencoder.eval()
                    best_auc = auc
                    best_auc_logi = logi_auc
                    best_auc_struc = struc_auc
                    best_iter = iteration

                    # save anomaly maps
                    self.test(test_set=self.test_set,iteration=iteration,save_img=True)

                    self.val(test_set=self.validation_loader,iteration=iteration,save_img=True)
                    # save best model
                    model_dict = {
                        'teacher': self.teacher,
                        'student': self.student,
                        'autoencoder': self.autoencoder,
                        'teacher_mean': self.teacher_mean,
                        'teacher_std': self.teacher_std,
                        'q_ae_end': self.q_ae_end,
                        'q_ae_start': self.q_ae_start,
                        'q_st_end': self.q_st_end,
                        'q_st_start': self.q_st_start,
                    }
                    torch.save(model_dict, os.path.join(config["ckpt_dir"], f'best_{config["category"]}.pth'))
                    # teacher frozen
                    self.teacher.eval()
                    self.student.train()
                    self.autoencoder.train()
    
        print(f'Best image auc of {config["category"]}: {best_auc:.4f}')
        return best_auc, best_iter, best_auc_logi, best_auc_struc


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.abspath(__file__))
    config = read_config("./configs/mvtecloco_train.yaml")
    categories = ['splicing_connectors','breakfast_box','juice_bottle','pushpins','screw_bag',]
    runs = 1
    seeds = [random.randint(0,1000) for i in range(runs)]
    run_name = "LGST"
    for c in categories:
        best_auroc_list = []
        best_iter_list = []
        best_auroc_list_logi = []
        best_auroc_list_struc = []
        for i in range(runs):
            config = read_config("./configs/mvtecloco_train.yaml")
            config['category'] = c
            config['seed'] = seeds[i]
            config['run_name'] = run_name
            csad = CSAD(config)
            best_auroc, best_iter, best_auc_logi, best_auc_struc = csad.train()
            best_auroc_list.append(best_auroc)
            best_iter_list.append(best_iter)
            best_auroc_list_logi.append(best_auc_logi)
            best_auroc_list_struc.append(best_auc_struc)
            with open(f'./results/{run_name}/best_auroc.txt','a') as f:
                f.write(f'{c}:\n')
                f.write(f'run{i}:\n')
                f.write(f'best_iter:{best_iter}\n')
                f.write(f'best_auc:{best_auroc:.4f}\n')
                f.write(f'best_auc_logi:{best_auc_logi:.4f}\n')
                f.write(f'best_auc_struc:{best_auc_struc:.4f}\n\n')
        best_auroc = np.mean(np.array(best_auroc_list))
        std_auroc = np.std(np.array(best_auroc_list))
        best_auc_logi = np.mean(np.array(best_auroc_list_logi))
        best_auc_struc = np.mean(np.array(best_auroc_list_struc))
        std_auroc_logi = np.std(np.array(best_auroc_list_logi))
        std_auroc_struc = np.std(np.array(best_auroc_list_struc))

        #best_auroc = 100
        with open(f'./results/{run_name}/best_auroc.txt','a') as f:
            f.write(f'total:\n')
            f.write(f'best_auc:{best_auroc:.4f}+-{std_auroc:.4f}\n')
            f.write(f'best_auc_logi:{best_auc_logi:.4f}+-{std_auroc_logi:.4f}\n')
            f.write(f'best_auc_struc:{best_auc_struc:.4f}+-{std_auroc_struc:.4f}\n\n')

