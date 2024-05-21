from __future__ import absolute_import, division, print_function
import os
from utils import utils as util
import numpy as np
import time
import json
import tqdm
import os
import cv2
from torchvision import transforms
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from collections import OrderedDict
torch.manual_seed(100)
torch.cuda.manual_seed(100)
from network.model_s3d import Panoformer as PanoBiT

import matplotlib.pyplot as plt
shuffled_cmap = np.load('utils/colormap.npy', allow_pickle=True).item()


class Inference:
    def __init__(self, settings):
        self.settings = settings
    
        print(self.settings)
        util.init_distributed_mode(self.settings)
        self.device = torch.device(self.settings.device)
        
        # Fix the seed for reproducibility
        seed = util.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = True
    
        num_tasks = util.get_world_size()
        global_rank = util.get_rank()
    

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        rgb = cv2.imread(self.settings.path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(1024, 512), interpolation=cv2.INTER_CUBIC)
        rgb = self.to_tensor(rgb.copy())
        self.inputs = rgb #self.normalize(rgb)
        self.settings.num_classes = 41
        self.model = PanoBiT(num_classes = self.settings.num_classes, target = "all")
        self.model.to(self.device)
        model_without_ddp = self.model
        if self.settings.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[self.settings.gpu], find_unused_parameters=True)
            model_without_ddp = self.model.module
        
        if self.settings.load_weights_dir is not None:
            self.load_model()
        ## Print Parameters 
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total number of parameters: {total_params}")
        print("Training is using:\n ", self.device)
    
    def process_batch(self, inputs, is_training = True):
        
            losses = {}
        
            equi_inputs = inputs.to(self.device)
        
            outputs = self.model(equi_inputs)
            # Depth loss
            # inputs["depth"] = inputs["depth"] * inputs["depth_mask"]
            outputs["pred_depth"] = outputs["pred_depth"]
            outputs["pred_semantic"] = outputs["pred_semantic"]
            outputs["pred_shading"] = outputs["pred_shading"]
            outputs["pred_albedo"] = outputs["pred_albedo"]
            outputs["pred_normal"] = outputs["pred_normal"]
            
            return outputs

    def validate(self):
        """Validate the model on the validation set
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.process_batch(self.inputs.unsqueeze(0), False)
            outputs["pred_shading"] = torch.clamp(outputs["pred_shading"],0,1) 
            outputs["pred_albedo"] = torch.clamp(outputs["pred_albedo"],0,1) 
            outputs["pred_normal"] = torch.clamp(outputs["pred_normal"],0,1) 
            outputs["pred_semantic"] = F.softmax(outputs["pred_semantic"], dim=1)
            outputs["pred_semantic"] = torch.argmax(outputs["pred_semantic"], dim=1)
            # print(outputs["pred_albedo"].shape)
            # Specify the path where you want to create the directory
            directory_path = 'output'
            
            try:
                os.mkdir(directory_path)
                print(f"Directory '{directory_path}' created successfully")
            except FileExistsError:
                print(f"Directory '{directory_path}' already exists")
            except OSError as error:
                print(f"Error creating directory '{directory_path}': {error}")
            plt.imsave(f"output/depth.jpg", 
                       outputs['pred_depth'][0].data.squeeze().cpu().numpy(), cmap = "plasma")
            plt.imsave(f"output/shading.jpg", 
                       outputs['pred_shading'][0].data.squeeze().cpu().numpy(), cmap = "gray")
            plt.imsave(f"output/albedo.jpg", 
                       outputs['pred_albedo'][0].permute(1, 2, 0).data.squeeze().cpu().numpy())
            plt.imsave(f"output/normal.jpg", 
                       outputs['pred_normal'][0].permute(1, 2, 0).data.squeeze().cpu().numpy())
            plt.imsave(f"output/semantic.jpg", 
                       outputs['pred_semantic'][0].cpu().numpy(), cmap = shuffled_cmap)


    def load_model(self):
        """Load model from disk
        """
        self.settings.load_weights_dir = os.path.expanduser(self.settings.load_weights_dir)

        assert os.path.isdir(self.settings.load_weights_dir), \
            "Cannot find folder {}".format(self.settings.load_weights_dir)
        print("loading model from folder {}".format(self.settings.load_weights_dir))

        path = os.path.join(self.settings.load_weights_dir, "{}.pth".format("model"))
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(path, map_location = "cuda:0")
        if not self.settings.distributed:
            pretrained_dict = OrderedDict((key.replace('module.', '', 1) \
                                           if key.startswith('module.') else key, value) \
                                          for key, value in pretrained_dict.items())
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        del pretrained_dict, model_dict


def get_args_parser():
    parser = argparse.ArgumentParser(
        'MultiPanoWise training and evaluation script', add_help=False)
    # Model parameters
    parser.add_argument('--model_name', default='MPW', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=[512, 1024],
                        type=int, help='images input size')
    parser.add_argument('--path', default='Data/full', type=str, metavar='DATASET PATH',
                        help='Path to dataset')


    parser.add_argument("--load_weights_dir", default=None, type=str, help="folder of model to load")
    
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--log_dir", type=str, default=os.path.join(os.path.dirname(__file__), "tmp"), help="log directory")
    parser.add_argument("--log_frequency", type=int, default=100, help="number of batches between each tensorboard log")
    parser.add_argument("--save_frequency", type=int, default=1, help="number of epochs between each save")
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    return parser

def main(args):
    infere = Inference(args)
    infere.validate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'MultiPanoWise training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

        