from __future__ import absolute_import, division, print_function
import os
import comet_ml
from utils import utils as util
import numpy as np
import time
import json
import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
torch.manual_seed(100)
torch.cuda.manual_seed(100)
from utils.metrics_s2d3d import compute_depth_metrics, Evaluator, compute_semantic_metrics
from utils.metrics_s2d3d import mIoU
import utils.loss_gradient as loss_g
import utils.losses as loss
from utils.losses import BerhuLoss
from network.model_s2d3d import Panoformer as PanoBiT
from utils.stanford2d3d import Stanford2d3d
from collections import OrderedDict
import matplotlib.pyplot as plt
shuffled_cmap = np.load('utils/colormap.npy', allow_pickle=True).item()
# Helper function to display logged assets in the Comet UI
def display(tab=None):
    experiment = comet_ml.get_global_experiment()
    experiment.display(tab=tab)


semantic_loss = nn.CrossEntropyLoss(reduction='mean')
ssim = loss.SSIMLoss()

def gradient(x):
    gradient_model = loss_g.Gradient_Net()
    g_x, g_y = gradient_model(x)
    return g_x, g_y

def gradient3d(x):
    gradient_model = loss_g.Gradient_Net_3d()
    g_x, g_y = gradient_model(x)
    return g_x, g_y

compute_loss = BerhuLoss()

def loss_behru(gt, pred, input, output):
    G_x, G_y = gradient(gt.float())
    p_x, p_y = gradient(pred)
    loss =compute_loss(input.float(), output) +\
                         compute_loss(G_x, p_x) +\
                         compute_loss(G_y, p_y)
    return loss

def loss_behru3d(gt, pred, input, output):
    G_x, G_y = gradient3d(gt.float())
    p_x, p_y = gradient3d(pred)
    loss =compute_loss(input.float(), output) +\
                         compute_loss(G_x, p_x) +\
                         compute_loss(G_y, p_y)
    return loss
def semantic_loss1(semantic, pred):
    ssloss = semantic_loss(pred, semantic) + loss.dice_coefficient_loss(semantic, pred)
    return ssloss



class Trainer:
    def __init__(self, settings):
        self.settings = settings
        comet_init_config = {
            "api_key":"QNHPKSygOyiOMg3t2DYAE1rBq",
            "project_name": self.settings.model_name,
            "workspace": "uzshah"
        }
        if util.get_rank() == 0:
            comet_ml.init(api_key=comet_init_config['api_key'],  
                          workspace=comet_init_config['workspace'],
                          project_name=comet_init_config['project_name']
                         )
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
    
        self.log_path = os.path.join(self.settings.log_dir, self.settings.model_name)
        train_dataset = Stanford2d3d(self.settings.data_path, "", num_classes = 14, 
                                     height=settings.input_size[0], width=settings.input_size[1],
                                      disable_color_augmentation=True, disable_LR_filp_augmentation=True, 
                                     disable_yaw_rotation_augmentation=True, is_training = True)
        val_dataset = Stanford2d3d(self.settings.data_path, "", num_classes = 14, 
                                   height=settings.input_size[0], width=settings.input_size[1],
                                    disable_color_augmentation=True, disable_LR_filp_augmentation=True, 
                                    disable_yaw_rotation_augmentation=True, is_training = False)
            
        sampler_train = torch.utils.data.DistributedSampler(
                train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        self.train_loader = DataLoader(train_dataset, self.settings.batch_size,  sampler= sampler_train, 
                                       num_workers=self.settings.num_workers, pin_memory=self.settings.pin_mem, drop_last=True)
        # torch.autograd.set_detect_anomaly(True)
        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // self.settings.batch_size * self.settings.num_epochs
        
        sampler_test = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        self.val_loader = DataLoader(val_dataset, self.settings.batch_size,  sampler= sampler_test,
                                     num_workers=self.settings.num_workers, pin_memory=self.settings.pin_mem, drop_last=True)
        self.settings.num_classes = 14
        self.model = PanoBiT(num_classes = self.settings.num_classes)
        self.model.to(self.device)
        model_without_ddp = self.model
        if self.settings.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[self.settings.gpu], find_unused_parameters=True)
            model_without_ddp = self.model.module
        self.parameters_to_train = list(self.model.parameters())

        self.optimizer = optim.Adam(self.parameters_to_train,
                                    self.settings.learning_rate)
        
        if self.settings.load_weights_dir is not None:
            self.load_model()

        ## Print Parameters 
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total number of parameters: {total_params}")
        print("Training model named:\n ", self.settings.model_name)
        print("Models and tensorboard events files are saved to:\n", self.settings.log_dir)
        print("Training is using:\n ", self.device)
        self.evaluator = Evaluator(self.settings)

        self.writers = {}
        for mode in ["train", "val"]:
            if util.get_rank() == 0 :
                self.writers[mode] = SummaryWriter(os.path.join(self.log_path,mode),comet_config={"disabled": False})
            else:
                self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))
    

        self.save_settings()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        self.validate()
        for self.epoch in range(self.settings.start_epoch, self.settings.num_epochs):
            self.train_one_epoch()
            self.validate()
            if (self.epoch + 1) % self.settings.save_frequency == 0:
                self.save_model()

    def train_one_epoch(self):
        """Run a single epoch of training
        """
        self.model.train()

        pbar = tqdm.tqdm(self.train_loader)
        pbar.set_description("Training Epoch_{}".format(self.epoch))
        m_w = 0
        v_loss = 0
        for batch_idx, inputs in enumerate(pbar):

            outputs, losses = self.process_batch(inputs, True)
            v_loss +=losses["loss"]
            pbar.set_postfix({'TL': v_loss.item()/(1+batch_idx)}, refresh=True)
            if self.settings.weighted_loss:
                losses_stacked = torch.stack([losses["depth_loss"], losses["normal_loss"], losses["semantic_loss"]])
                multitaskloss, m_w = self.MultiTaskLoss_instance(losses_stacked)
            else:
                multitaskloss = losses["loss"]
            self.optimizer.zero_grad(),
            multitaskloss.backward()
            self.optimizer.step()

            # log less frequently after the first 1000 steps to save time & disk space
            early_phase = batch_idx % self.settings.log_frequency == 0 and self.step < 500
            late_phase = self.step % 500 == 0
            errors = []
            if early_phase or late_phase:
                errors.extend(compute_depth_metrics(inputs["depth"].detach(), outputs["pred_depth"].detach(), inputs["depth_mask"]))
                errors.extend(compute_semantic_metrics(inputs["semantic"].detach(), outputs["pred_semantic"].detach()))
                
                for i, key in enumerate(self.evaluator.metrics.keys()):
                    losses[key] = np.array(errors[i].cpu())
                self.log("train", inputs, outputs, losses)

            self.step += 1
        print(f"final weights for the losses {m_w}")
    def process_batch(self, inputs, is_training = True):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
    
        losses = {}
    
        equi_inputs = inputs["rgb"]*inputs["val_mask"]
    
        outputs = self.model(equi_inputs)
        # Depth loss
        inputs["depth"] = inputs["depth"] * inputs["depth_mask"]
        outputs["pred_depth"] = outputs["pred_depth"] * inputs["depth_mask"]
        losses["depth_loss"] = loss_behru(inputs["depth"].float(), outputs["pred_depth"], \
                                          inputs["depth"].float(), outputs["pred_depth"])
    
        ## semantic loss
        inputs["semantic"]  = inputs["semantic"] * inputs["val_mask"]
        outputs["pred_semantic"] = outputs["pred_semantic"] * inputs["val_mask"]
        losses['semantic_loss'] = semantic_loss1(inputs["semantic"], outputs["pred_semantic"])
        losses['mIoUloss'] = 1 - mIoU(inputs["semantic"], outputs["pred_semantic"])
        if is_training:
            losses['loss'] = self.settings.alpha*torch.max(losses["depth_loss"], losses['semantic_loss']) +\
            (1-self.settings.alpha)*(losses["depth_loss"]+losses['semantic_loss'])
        else:
            losses['loss'] = losses['mIoUloss'] + losses["depth_loss"]
        
        return outputs, losses

    def validate(self):
        """Validate the model on the validation set
        """
        self.model.eval()

        self.evaluator.reset_eval_metrics()

        pbar = tqdm.tqdm(self.val_loader)
        pbar.set_description("Validating Epoch_{}".format(self.epoch))
        counter = 0
        v_loss = 0
        with torch.no_grad():
            for batch_idx, inputs in enumerate(pbar):
                outputs, losses = self.process_batch(inputs, False)
                v_loss +=losses["loss"]
                pbar.set_postfix({'TL': v_loss.item()/(1+batch_idx)}, refresh=True)
                # print(losses["loss"], inputs["path"])
                if self.settings.target == "all":
                    self.evaluator.compute_eval_metrics(inputs["depth"].detach(), outputs["pred_depth"].detach(), \
                                               inputs["semantic"].detach(), outputs["pred_semantic"].detach(), \
                                               dmask=inputs["depth_mask"], mask = inputs["val_mask"])
                if self.settings.target == "depth":
                   self.evaluator.compute_eval_metrics(gt_depth=inputs["depth"].detach(), 
                                                       pred_depth=outputs["pred_depth"].detach(), \
                                               dmask=inputs["depth_mask"], mask = inputs["val_mask"]) 
                if self.settings.target == "semantic":
                    
                   self.evaluator.compute_eval_metrics(gt_semantic=inputs["semantic"].detach(), 
                                                       pred_semantic=outputs["pred_semantic"].detach(), \
                                                       mask = inputs["val_mask"]) 
                if counter%10==0 and self.settings.num_epochs ==0:
                    self.log("val", inputs, outputs, losses, counter)
                counter +=1
        self.evaluator.print()
        for i, key in enumerate(self.evaluator.metrics.keys()):
            # print(key)
            losses[key] = np.array(self.evaluator.metrics[key].avg.cpu())
        self.log("val", inputs, outputs, losses, batch_idx)
        
        del inputs, outputs, losses

    def log(self, mode, inputs, outputs, losses, batch_idx=0):
        """Write an event to the tensorboard events file
        """
        if self.settings.target == "semantic" or self.settings.target=="all":
            outputs["pred_semantic"] = F.softmax(outputs["pred_semantic"], dim=1)
            outputs["pred_semantic"] = torch.argmax(outputs["pred_semantic"], dim=1)
            inputs["semantic"] = torch.argmax(inputs["semantic"], dim=1)
        
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(2, self.settings.batch_size)):  # write a maxmimum of four images
            writer.add_image("rgb/{}-{}".format(j, batch_idx), inputs["rgb"][j].data, self.step)
            if self.settings.target == "all" or self.settings.target == "depth":
                fig_target = plt.figure(0,figsize=(8,4),dpi=128,layout='tight')
                plt.axis('off')
                
                plt.imshow(inputs["depth"][j].data.squeeze().cpu().numpy(), cmap='plasma')
                writer.add_figure("gt_depth/{}-{}".format(j, batch_idx), fig_target ,self.step)
                 
                fig_target = plt.figure(0,figsize=(8,4),dpi=128,layout='tight')
                plt.axis('off')                 
                plt.imshow(outputs["pred_depth"][j].data.squeeze().cpu().numpy(), cmap='plasma')
                writer.add_figure("pred_depth/{}-{}".format(j, batch_idx), fig_target ,self.step)
        

            if self.settings.target == "all" or self.settings.target == "semantic":
                fig_target = plt.figure(0,figsize=(8,4),dpi=128,layout='tight')
                plt.axis('off')
                plt.imshow(inputs["semantic"][j].cpu().numpy(), cmap = shuffled_cmap)
                writer.add_figure("gt_semantic/{}-{}".format(j,batch_idx), fig_target, self.step)
                fig_pred = plt.figure(0,figsize=(8,4),dpi=128,layout='tight')
                plt.axis('off')
                plt.imshow(outputs["pred_semantic"][j].cpu().numpy(), cmap = shuffled_cmap)
                writer.add_figure("pred_semantic/{}-{}".format(j,batch_idx), fig_pred, self.step)
            
            
    def save_settings(self):
        """Save settings to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.settings.__dict__.copy()

        with open(os.path.join(models_dir, 'settings.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, "{}.pth".format("model"))
        to_save = self.model.state_dict()
        torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.optimizer.state_dict(), save_path)

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
        pretrained_dict["module.context_se.output.weight"] = pretrained_dict["module.context_se.output.weight"][:14,:,:,:]
        pretrained_dict["module.context_se.output.bias"] = pretrained_dict["module.context_se.output.bias"][:14]
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        del pretrained_dict, model_dict
        # loading adam state
        optimizer_load_path = os.path.join(self.settings.load_weights_dir, "adam1.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

