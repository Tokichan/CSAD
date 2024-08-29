import cv2
import numpy as np
import glob
import random
import albumentations as A
import matplotlib.pyplot as plt


color_list = [[127, 123, 229], [195, 240, 251], [120, 200, 255],
               [243, 241, 230], [224, 190, 144], [178, 116, 75],
               [255, 100, 0], [0, 255, 100],
              [100, 0, 255], [100, 255, 0], [255, 0, 255],
              [0, 255, 255], [192, 192, 192], [128, 128, 128],
              [128, 0, 0], [128, 128, 0], [0, 128, 0],
              [128, 0, 128], [0, 128, 128], [0, 0, 128]]
        
def turn_binary_to_int(mask):
    temp = np.zeros_like(mask,dtype=np.uint8)
    temp[mask]=255
    return temp

def intersect_ratio(mask1,mask2):
    intersection = np.logical_and(mask1,mask2)
    if intersection.sum() == 0:
        return 0
    ratio = np.sum(intersection)/min([np.sum(mask1!=0),np.sum(mask2!=0)])
    ratio = 0 if np.isnan(ratio) else ratio
    return ratio

def split_masks_from_one_mask(masks):
    result_masks = list()
    for i in range(1,np.max(masks)+1):
        mask = np.zeros_like(masks)
        mask[masks==i] = 255
        if np.sum(mask!=0)/mask.size > 0.001:
            result_masks.append(mask)
    return result_masks

def sample_point(in_x, in_y,min_distance=0.4,boundary=0.05):
    # Number of points to sample
    assert min_distance < np.sqrt(2*(0.5-boundary)**2), "min_distance should be smaller than np.sqrt(2*(0.5-boundary)**2)"
    num_samples = 10000
    
    # Generate random points
    random_points = np.random.rand(num_samples, 2)
    random_points = random_points*(1-boundary*2) + boundary
    
    # Calculate distances from the input point
    distances = np.sqrt((random_points[:, 0] - in_x)**2 + (random_points[:, 1] - in_y)**2)
    
    # Calculate distances as weights
    distances[distances < min_distance] = 0
    weights = distances**2
    
    
    # Normalize weights to make them probabilities
    probabilities = weights / np.sum(weights)
    
    # Sample a point based on the probabilities
    sampled_index = np.random.choice(num_samples, p=probabilities)
    sampled_point = random_points[sampled_index, :]
    
    return sampled_point
    



        
def sample_mask(masks,weight_power=0.5):
    mask_weight = np.array([np.sum(m) for m in masks])
    mask_weight = mask_weight**weight_power
    # max_mask = np.argmax(mask_weight)
    # mask_weight[max_mask] = 0 # remove the largest mask
    mask_weight = mask_weight/np.sum(mask_weight)
    idx = np.random.choice(np.arange(len(masks)),p=mask_weight)
    source_mask = masks[idx]
    return source_mask, idx

def labeled_lsa(source_img,source_masks,source_labelmap,target_img,target_masks,target_labelmap,target_background,config):
    # both source and target masks are labeled
    # paste source image to target image
    source_masks = split_masks_from_one_mask(source_masks)
    # filter masks that are background
    new_source_masks = list()
    for m in source_masks:
        # cv2.imshow(f"{intersect_ratio(m,source_labelmap)}",np.hstack([m,source_labelmap*30]))
        # cv2.waitKey(0)
        if intersect_ratio(m,source_labelmap) > 0.9:
            new_source_masks.append(m)
    source_masks = new_source_masks

    aug_num = 0
    final_mask = np.zeros_like(target_masks)
    result_img = target_img.copy()
    result_labelmap = target_labelmap.copy()
    while aug_num < config['min_aug_num']:
        source_mask,idx = sample_mask(source_masks,config['weight_power'])
        source_masks.pop(idx)
        if (source_labelmap[source_mask>0]==0).sum() > 100:
            continue
        for attempt in range(500):
            if attempt+1 % 10 == 0:
                # re-sample source mask
                source_mask,idx = sample_mask(source_masks,config['weight_power'])
                source_masks.pop(idx)
                if (source_labelmap[source_mask>0]==0).sum() > 100:
                    continue
            bbox = cv2.boundingRect(source_mask)
            source_x = bbox[0]+bbox[2]//2
            source_y = bbox[1]+bbox[3]//2
            target_point = sample_point(source_x/target_img.shape[1],source_y/target_img.shape[0],
                                    min_distance=config['min_distance'],
                                    boundary=config['boundary'])
            target_x = target_point[0]*target_img.shape[1]
            target_y = target_point[1]*target_img.shape[0]
            delta_x = target_x-source_x
            delta_y = target_y-source_y
            rotate_matrix = cv2.getRotationMatrix2D([source_x,source_y], np.random.randint(0,360), 1.0)
            rotate_matrix = np.vstack([rotate_matrix,np.array([0,0,1])])
            translation_matrix = np.array([[1,0,delta_x],[0,1,delta_y]])
            translation_matrix = np.vstack([translation_matrix,np.array([0,0,1])])
            affine_matrix = np.dot(translation_matrix,rotate_matrix)
            affine_matrix = affine_matrix[:2]
            result_mask = cv2.warpAffine(source_mask,
                                            affine_matrix,
                                            (source_mask.shape[1],source_mask.shape[0]))
            result_mask[result_mask>200] = 255
            result_mask[result_mask<=200] = 0

            if intersect_ratio(result_mask,target_background) < 0.5:
                # found a proper position
                # print(f"intersect ratio:{intersect_ratio(result_mask,target_background)}")
                break
        # print(f"attempt:{attempt}")
        if attempt >= 500:
            # failed to find a proper position
            continue
        else:
            temp_img = np.zeros_like(source_img)
            temp_img[source_mask>0] = source_img[source_mask>0]
            temp_img = cv2.warpAffine(temp_img,
                                    affine_matrix,
                                    (source_mask.shape[1],source_mask.shape[0]))
            temp_labelmap = np.zeros_like(source_labelmap)
            temp_labelmap[source_mask>0] = source_labelmap[source_mask>0]
            temp_labelmap = cv2.warpAffine(temp_labelmap,
                                    affine_matrix,
                                    (source_mask.shape[1],source_mask.shape[0]))
            result_labelmap[result_mask>0] = temp_labelmap[result_mask>0]
            result_img[result_mask>0] = temp_img[result_mask>0]
            final_mask[result_mask>0] = 255
            aug_num += 1

    # visualize
    # vis_image = np.hstack([source_img,cv2.cvtColor(source_mask,cv2.COLOR_GRAY2BGR),result_img,cv2.cvtColor(final_mask,cv2.COLOR_GRAY2BGR)])
    # vis_image = cv2.cvtColor(vis_image,cv2.COLOR_RGB2BGR)
    # cv2.imshow("result",vis_image)
    # cv2.waitKey(0)
            
    for i in range(config['max_aug_num']-config['min_aug_num']):
        source_mask,idx = sample_mask(source_masks,config['weight_power'])
        source_masks.pop(idx)
        if (source_labelmap[source_mask>0]==0).sum() > 100:
            continue
        for attempt in range(500+1):
            if attempt+1 % 10 == 0:
                # re-sample source mask
                source_mask,idx = sample_mask(source_masks,config['weight_power'])
                source_masks.pop(idx)
                if (source_labelmap[source_mask>0]==0).sum() > 100:
                    continue
            bbox = cv2.boundingRect(source_mask)
            source_x = bbox[0]+bbox[2]//2
            source_y = bbox[1]+bbox[3]//2
            target_point = sample_point(source_x/target_img.shape[1],source_y/target_img.shape[0],
                                    min_distance=config['min_distance'],
                                    boundary=config['boundary'])
            target_x = target_point[0]*target_img.shape[1]
            target_y = target_point[1]*target_img.shape[0]
            delta_x = target_x-source_x
            delta_y = target_y-source_y
            rotate_matrix = cv2.getRotationMatrix2D([source_x,source_y], np.random.randint(0,360), 1.0)
            rotate_matrix = np.vstack([rotate_matrix,np.array([0,0,1])])
            translation_matrix = np.array([[1,0,delta_x],[0,1,delta_y]])
            translation_matrix = np.vstack([translation_matrix,np.array([0,0,1])])
            affine_matrix = np.dot(translation_matrix,rotate_matrix)
            affine_matrix = affine_matrix[:2]
            result_mask = cv2.warpAffine(source_mask,
                                            affine_matrix,
                                            (source_mask.shape[1],source_mask.shape[0]))
            result_mask[result_mask>200] = 255
            result_mask[result_mask<=200] = 0

            if intersect_ratio(result_mask,target_background) < 0.9:
                # found a proper position
                break

        # print(f"attempt:{attempt}")
        if attempt < 500:
            temp_img = np.zeros_like(source_img)
            temp_img[source_mask>0] = source_img[source_mask>0]
            temp_img = cv2.warpAffine(temp_img,
                                    affine_matrix,
                                    (source_mask.shape[1],source_mask.shape[0]))
            temp_labelmap = np.zeros_like(source_labelmap)
            temp_labelmap[source_mask>0] = source_labelmap[source_mask>0]
            temp_labelmap = cv2.warpAffine(temp_labelmap,
                                    affine_matrix,
                                    (source_mask.shape[1],source_mask.shape[0]))
            result_labelmap[result_mask>0] = temp_labelmap[result_mask>0]
            result_img[result_mask>0] = temp_img[result_mask>0]
            final_mask[result_mask>0] = 255
            
            aug_num += 1
    ###############################
    # visualize
    # color_labelmap = np.zeros_like(result_img)
    # for i in range(1,np.max(result_labelmap)+1):
    #     color_labelmap[result_labelmap==i] = color_list[i-1][::-1]
    # vis_image = np.hstack([result_img,cv2.cvtColor(final_mask,cv2.COLOR_GRAY2BGR),color_labelmap])
    # vis_image = cv2.cvtColor(vis_image,cv2.COLOR_RGB2BGR)

    # cv2.imshow("result",vis_image)
    # cv2.waitKey(0)
    #cv2.imwrite("result.png",vis_image)

    return result_img, result_labelmap#, vis_image


class LabeledLSA():
    def __init__(self,images,masks,label_maps,backgrounds,config):
        self.images = images
        self.masks = masks
        self.label_maps = label_maps
        self.backgrounds = backgrounds
        self.config = config
    
    def augment(self,idx):
        target_img = self.images[idx]
        target_masks = self.masks[idx]
        target_background = self.backgrounds[idx]
        target_labelmap = self.label_maps[idx]

        source_idx = idx#random.choice(np.arange(len(self.images)))
        source_img = self.images[idx]
        source_masks = self.masks[idx]
        source_labelmap = self.label_maps[idx]
        # source_background = self.backgrounds[source_idx]

        result_img, result_labelmap = labeled_lsa(source_img,
                                               source_masks,
                                               source_labelmap,
                                               target_img,
                                               target_masks,
                                               target_labelmap,
                                               target_background,
                                               self.config)

        return result_img, result_labelmap
    



