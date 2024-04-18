from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Function to calculate the angular error between two vectors
def angular_error(n_gt, n_hat):
    # n_gt, n_hat = n_gt.cuda(), n_hat.cuda()
    # Convert to PyTorch tensor if input is a NumPy array
    if isinstance(n_gt, np.ndarray):
        n_gt = torch.from_numpy(n_gt)
    if isinstance(n_hat, np.ndarray):
        n_hat = torch.from_numpy(n_hat)

    n_gt = 2.0*n_gt-1
    # Normalizing the vectors to avoid division by zero errors in case of non-normalized inputs
    n_gt_norm = F.normalize(n_gt, p=2, dim=2)
    n_hat_norm = F.normalize(n_hat, p=2, dim=2)

    # Calculate the cosine of the angular error
    w_hat = torch.sum(n_hat_norm * n_hat_norm, dim=2) !=1.0 
    w_gt = torch.sum(n_gt_norm * n_gt_norm, dim=2) !=1.0 
    w = w_hat * w_gt
    
    cos_angle_error = torch.clamp(torch.sum(n_gt_norm * n_hat_norm, dim=2), -1.0, 1.0)

    # Calculate the angle in radians and then convert to degrees
    angle_error_rad = (torch.acos(cos_angle_error) * w)/torch.pi
    angle_error_rad = torch.abs(angle_error_rad)
    # angle_error_deg = torch.rad2deg(angle_error_rad)

    return angle_error_rad

# Function to calculate the Average Angular Error (AAE)
def average_angular_error(n_gt, n_hat):
    n_gt, n_hat = n_gt.cuda(), n_hat.cuda()
    angle_error_deg = angular_error(n_gt, n_hat)
    aae = torch.mean(angle_error_deg.float())
    return aae

# Function to calculate the Proportion of Good Pixels (PGP)
def proportion_good_pixels(n_gt, n_hat, phi=20):
    n_gt, n_hat = n_gt.cuda(), n_hat.cuda()
    angle_error_deg = angular_error(n_gt, n_hat)
    pgp = torch.mean((angle_error_deg <= phi).float())
    return 1-pgp



torch.manual_seed(10)
torch.cuda.manual_seed(10)


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, target, pred, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        valid_mask = (target > 0).detach()

        if mask is not None:
            valid_mask *= mask.detach()

        diff = target - pred
        diff = diff[valid_mask]
        loss = diff.abs().mean()
        return loss


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, target, pred, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        if mask is not None:
            valid_mask *= mask.detach()

        diff = target - pred
        diff = diff[valid_mask]
        loss = (diff**2).mean()
        return loss


class BerhuLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerhuLoss, self).__init__()
        self.threshold = threshold

    def forward(self, target, pred, mask=None, d_map=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        #valid_mask = (target > 0).detach()
        #if mask is not None:
            #valid_mask *= mask.detach()

        diff = torch.abs(target - pred)
        #diff = diff[valid_mask]
        #d_map = d_map[valid_mask]
        delta = self.threshold * torch.max(diff).data.cpu().numpy()

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff ** 2 + delta ** 2, 2.0*delta ** 2, 0.)
        part2 = part2 / (2. * delta)
        diff = part1 + part2
        loss = diff.mean()
        return loss



class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()

        self.window_size = window_size
        self.size_average = size_average

        K1 = 0.01
        K2 = 0.03
        self.C1 = (K1 * 255) ** 2
        self.C2 = (K2 * 255) ** 2

    def forward(self, img1, img2):
        mu1 = F.avg_pool2d(img1, self.window_size, stride=1, padding=0)
        mu2 = F.avg_pool2d(img2, self.window_size, stride=1, padding=0)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(img1 ** 2, self.window_size, stride=1, padding=0) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 ** 2, self.window_size, stride=1, padding=0) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, self.window_size, stride=1, padding=0) - mu1_mu2

        SSIM_n = (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        SSIM_d = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)

        ssim_index = SSIM_n / SSIM_d

        if self.size_average:
            return 1 - ssim_index.mean(dim=(1, 2, 3))
        else:
            return 1 - ssim_index.mean(1).mean(1).mean(1)

def compute_depth_metrics(gt, pred, min, max, mask=None, median_align=True):
    """Computation of metrics between predicted and ground truth depths
    """
    gt_depth = gt
    pred_depth = pred

    gt_depth[gt_depth<=min] = min
    pred_depth[pred_depth<=min] = min
    gt_depth[gt_depth >= max] = max
    pred_depth[pred_depth >= max] = max


    ###########STEP 1: compute delta#######################
    thresh = torch.max((gt_depth / pred_depth), (pred_depth / gt_depth))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    ##########STEP 2:compute mean error###################

    rmse = (gt_depth - pred_depth) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log10(gt_depth) - torch.log10(pred_depth)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_ = torch.mean(torch.abs(gt_depth - pred_depth))

    abs_rel = torch.mean(torch.abs(gt_depth - pred_depth) / gt_depth)

    sq_rel = torch.mean((gt_depth - pred_depth) ** 2 / gt_depth)

    log10 = torch.mean(torch.abs(torch.log10(pred / gt_depth)))

    mae = torch.mean(torch.abs((pred_depth - gt_depth)) / gt_depth)
    
    mre = torch.mean(((pred_depth - gt_depth)** 2) / gt_depth)
    #mae = (gt_depth - pred_depth).abs().mean()
    #mre = ((gt_depth - pred_depth).abs() / gt).mean()

    return mre.item(), mae.item(), abs_.item(), abs_rel.item(), sq_rel.item(), rmse.item(), rmse_log.item(), log10.item(), a1.item(), a2.item(), a3.item()


def Dice_Loss(pred, target, smooth=1.0):
    """
    Calculate the multi-class Dice loss.

    Args:
        predictions (torch.Tensor): Predicted class probabilities with shape (batch_size, num_classes, ...)
        targets (torch.Tensor): Ground truth class depth with shape (batch_size, ..., ...)
        num_classes (int): Number of classes
        epsilon (float): Small constant to avoid division by zero

    Returns:
        torch.Tensor: Dice loss
    """
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=151):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = torch.argmax(mask, dim=1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)

def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        mask = torch.argmax(mask, dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_pred = F.softmax(y_pred, dim=1)
    intersection = torch.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)

def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)
