import cv2
import numpy as np
import random
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
import os


def read_list(txt_file_path):
    # Read the content of the text file
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
    
    # Process each line and generate the list of filenames
    file_list = []
    for line in lines:
        # Split the line based on space
        parts = line.strip().split()
    
        # Combine the parts to form the filename in the desired format
        if len(parts) == 2:
            filename = f"{parts[0]}_{parts[1]}.png"
            file_list.append(filename)
    return file_list


class Structured3D(data.Dataset):
    """The Structured3D Dataset"""

    def __init__(self, root_dir, list_file, num_classes, height=512, width=1024, disable_color_augmentation=False,
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
        if is_training:
            self.rgb_depth_list = read_list(os.path.join(root_dir, 'train_clean.txt'))
            # self.rgb_depth_list = self.rgb_depth_list
        else:
            self.rgb_depth_list = read_list(os.path.join(root_dir, 'test.txt'))
        self.w = width
        self.h = height

        self.max_depth_meters = 10.0

        self.color_augmentation = False
        self.LR_filp_augmentation = False
        self.yaw_rotation_augmentation = False

        self.is_training = is_training


        if self.color_augmentation:
            try:
                self.brightness = (0.8, 1.2)
                self.contrast = (0.8, 1.2)
                self.saturation = (0.8, 1.2)
                self.hue = (-0.1, 0.1)
                self.color_aug= transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)
            except TypeError:
                self.brightness = 0.2
                self.contrast = 0.2
                self.saturation = 0.2
                self.hue = 0.1
                self.color_aug = transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def __len__(self):
        return len(self.rgb_depth_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}

        rgb_name = os.path.join(self.root_dir, "rgb",self.rgb_depth_list[idx])
        # print(rgb_name)
        rgb = cv2.imread(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)
        
        depth_name = os.path.join(self.root_dir, "depth", self.rgb_depth_list[idx])
        gt_depth = cv2.imread(depth_name, -1)
        gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        gt_depth = gt_depth.astype(float)/1000.0
        gt_depth = gt_depth.astype(np.float32)
        gt_depth[gt_depth > self.max_depth_meters] = self.max_depth_meters

        shading_name = os.path.join(self.root_dir, "shading", self.rgb_depth_list[idx])
        gt_shading = cv2.imread(shading_name, -1)
        gt_shading = gt_shading.astype(float)/65535.0
        gt_shading = gt_shading.astype(np.float32)
        gt_shading = cv2.resize(gt_shading, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)

        albedo_name = os.path.join(self.root_dir, "albedo", self.rgb_depth_list[idx])
        gt_albedo = cv2.imread(albedo_name)
        gt_albedo = cv2.cvtColor(gt_albedo, cv2.COLOR_BGR2RGB)
        gt_albedo = cv2.resize(gt_albedo, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)

        normal_name = os.path.join(self.root_dir, "normal", self.rgb_depth_list[idx])
        gt_normal = cv2.imread(normal_name)
        gt_normal = cv2.cvtColor(gt_normal, cv2.COLOR_BGR2RGB)
        gt_normal = cv2.resize(gt_normal, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)

        semantic_name = os.path.join(self.root_dir, "semantic", self.rgb_depth_list[idx])
        mask = Image.open(semantic_name)
        mask = cv2.resize(np.array(mask), dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        
        
        if self.is_training and self.yaw_rotation_augmentation:
            # random yaw rotation
            roll_idx = random.randint(0, self.w)
            rgb = np.roll(rgb, roll_idx, 1)
            gt_depth = np.roll(gt_depth, roll_idx, 1)
            gt_normal = np.roll(gt_normal, roll_idx, 1)
            gt_shading = np.roll(gt_shading, roll_idx, 1)
            gt_albedo = np.roll(gt_albedo, roll_idx, 1)
            gt_semantic = np.roll(np.array(gt_semantic), roll_idx, 1)

        if self.is_training and self.LR_filp_augmentation and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            gt_depth = cv2.flip(gt_depth, 1)
            gt_normal = cv2.flip(gt_normal, 1)
            gt_shading = cv2.flip(gt_shading, 1)
            gt_albedo = cv2.flip(gt_albedo, 1)
            mask = cv2.flip(mask, 1)
            
        if self.is_training and self.color_augmentation and random.random() > 0.5:
            aug_rgb = np.asarray(self.color_aug(transforms.ToPILImage()(Image.fromarray(rgb))))
        else:
            aug_rgb = rgb

        #cube_rgb, cube_gt_depth = self.e2c.run(rgb, gt_depth[..., np.newaxis])
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        one_hot_mask = torch.zeros(self.num_classes, *mask.shape, dtype=torch.float32)
        gt_semantic = one_hot_mask.scatter_(0, mask.unsqueeze(0), 1)
        rgb = self.to_tensor(rgb.copy())
        aug_rgb = self.to_tensor(aug_rgb.copy())
        gt_albedo = self.to_tensor(gt_albedo.copy())
        gt_normal = self.to_tensor(gt_normal.copy())

        inputs["rgb"] = rgb
        inputs["normalized_rgb"] = self.normalize(aug_rgb)
        
        mask = torch.ones([1, self.h, self.w])
        inputs["depth"] = torch.from_numpy(np.expand_dims(gt_depth, axis=0))
        inputs["depth_mask"] = ((inputs["depth"] > 0) & (inputs["depth"] <= self.max_depth_meters)
                                & ~torch.isnan(inputs["depth"])) 
        inputs["depth_mask"] = inputs["depth_mask"] *mask
        inputs["albedo"] = gt_albedo
        inputs["normal"] = gt_normal
        inputs["semantic"] = gt_semantic if isinstance(gt_semantic, torch.Tensor) else torch.from_numpy(gt_semantic)
        inputs["shading"] = torch.from_numpy(np.expand_dims(gt_shading, axis=0))
            
                          
        
        inputs['val_mask'] = mask

        """
        cube_gt_depth = torch.from_numpy(np.expand_dims(cube_gt_depth[..., 0], axis=0))
        inputs["cube_gt_depth"] = cube_gt_depth
        inputs["cube_val_mask"] = ((cube_gt_depth > 0) & (cube_gt_depth <= self.max_depth_meters)
                                   & ~torch.isnan(cube_gt_depth))
        """

        return inputs
