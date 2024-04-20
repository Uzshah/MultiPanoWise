import cv2
import numpy as np
import random
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
import os, json

def read_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_depth_list.append(line.strip().split(" "))
    return rgb_depth_list


def _color2id(mask, img, id2label, num_classes = 14):
    mask = np.array(mask, np.int32)
    rgb = np.array(img, np.int32)
    unk = (mask[..., 0] != 0)
    mask = id2label[mask[..., 1] * 256 + mask[..., 2]]
    mask[unk] = 0
    mask[rgb.sum(-1) == 0] = 0
    mask -= 1  # 0->255
    mask[mask == 255] = num_classes - 1
    #print(np.unique(mask))
    return mask
    
class Stanford2d3d(data.Dataset):
    """The Stanford2d3d Dataset"""

    def __init__(self, root_dir, list_file, num_classes, height=256, width=512, disable_color_augmentation=False,
                 disable_LR_filp_augmentation=False, disable_yaw_rotation_augmentation=False, is_training=False):
        """
        Args:
            root_dir (string): Directory of the Stanford2D3D Dataset.
            list_file (string): Path to the txt file contain the list of image and depth files.
            height, width: input size.
            disable_color_augmentation, disable_LR_filp_augmentation,
            disable_yaw_rotation_augmentation: augmentation options.
            is_training (bool): True if the dataset is the training set.
        """
        self.root_dir = root_dir
        self.num_classes = num_classes 
        
        with open(os.path.join(self.root_dir, 'assets/semantic_labels.json'), 'r', encoding='utf8') as f:
            id2name = [name.split('_')[0] for name in json.load(f)] + ['<UNK>']
        with open(os.path.join(self.root_dir, 'assets/name2label.json'), 'r', encoding='utf8') as f:
            name2id = json.load(f)
        self.id2label = np.array([name2id[name] for name in id2name], np.uint8)
        if is_training:
            self.rgb_depth_list = read_list(os.path.join(root_dir, "train_full.txt"))
            self.rgb_depth_list = 3*self.rgb_depth_list
        else:
            self.rgb_depth_list = read_list(os.path.join(root_dir, "test_full.txt"))
        self.w = width
        self.h = height

        self.max_depth_meters = 10.0

        self.color_augmentation = False
        self.LR_filp_augmentation = True
        self.yaw_rotation_augmentation = True

        self.is_training = is_training


        if self.color_augmentation:
            self.color_aug = transforms.ColorJitter(
                    brightness=0.2, 
                    contrast=0.2, 
                    saturation=0.2, 
                    hue=0.1
                )


        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def __len__(self):
        return len(self.rgb_depth_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}

        rgb_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][0])
        # print(rgb_name)
        img_rgb = cv2.imread(rgb_name)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(img_rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)
#         rgb = rgb.astype(np.float32)/255.0
        
        depth_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][1])
        gt_depth = cv2.imread(depth_name, -1)
        gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        gt_depth = gt_depth.astype(float)/512.0
        gt_depth = gt_depth.astype(np.float32)
        gt_depth[gt_depth > self.max_depth_meters+1] = self.max_depth_meters + 1

        semantic_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][2])
        gt_mask = np.array(Image.open(semantic_name))
        mask = _color2id(gt_mask, img_rgb, self.id2label)
        mask = cv2.resize(mask, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        
        
        if self.is_training and self.yaw_rotation_augmentation:
            # random yaw rotation
            roll_idx = random.randint(0, self.w)
            rgb = np.roll(rgb, roll_idx, 1)
            gt_depth = np.roll(gt_depth, roll_idx, 1)
            mask = np.roll(np.array(mask), roll_idx, 1)

        if self.is_training and self.LR_filp_augmentation and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            gt_depth = cv2.flip(gt_depth, 1)
            mask = cv2.flip(mask, 1)
            
        if self.is_training and self.color_augmentation and random.random() > 0.5:
            aug_rgb = np.asarray(self.color_aug(transforms.ToPILImage()(Image.fromarray(rgb))))
        else:
            aug_rgb = rgb
        
        # transforms.ToPILImage()
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        one_hot_mask = torch.zeros(self.num_classes, *mask.shape, dtype=torch.float32)
        gt_semantic = one_hot_mask.scatter_(0, mask.unsqueeze(0), 1)
        
        if self.is_training and self.color_augmentation and random.random() > 0.5:
            aug_rgb = np.asarray(self.color_aug(Image.fromarray(rgb)))
        else:
            aug_rgb = rgb
            
        rgb = self.to_tensor(rgb.copy())
        aug_rgb = self.to_tensor(aug_rgb.copy())
        inputs["rgb"] = rgb
        inputs["normalized_rgb"] = self.normalize(aug_rgb)
        
        mask = torch.ones([1, self.h, self.w])
        mask[:, 0:int(self.h*0.15), :] = 0
        mask[:, self.h-int(self.h*0.15):self.h, :] = 0



        inputs["depth"] = torch.from_numpy(np.expand_dims(gt_depth, axis=0))
        inputs["semantic"] = gt_semantic if isinstance(gt_semantic, torch.Tensor) else torch.from_numpy(gt_semantic)
        
        inputs["depth_mask"] = ((inputs["depth"] > 0) & (inputs["depth"] <= self.max_depth_meters)
                                & ~torch.isnan(inputs["depth"]))
                                
        inputs["depth_mask"] = inputs["depth_mask"] * mask
        inputs['val_mask'] = mask
        
        """
        cube_gt_depth = torch.from_numpy(np.expand_dims(cube_gt_depth[..., 0], axis=0))
        inputs["cube_gt_depth"] = cube_gt_depth
        inputs["cube_val_mask"] = ((cube_gt_depth > 0) & (cube_gt_depth <= self.max_depth_meters)
                                   & ~torch.isnan(cube_gt_depth))
        """

        return inputs
