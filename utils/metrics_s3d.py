import os
import torch
import torch.nn.functional as F
import numpy as np
#==========================
# Depth Prediction Metrics
#==========================
def compute_depth_metrics(gt, pred, mask=None, median_align=True):
    """Computation of metrics between predicted and ground truth depths
    """
    gt_depth = gt
    pred_depth = pred

    gt_depth[gt_depth<=0.1] = 0.1
    pred_depth[pred_depth<=0.1] = 0.1
    gt_depth[gt_depth >= 11] = 11
    pred_depth[pred_depth >= 11] = 11


    ###########STEP 1: compute delta#######################
    thresh = torch.max((gt_depth / pred_depth), (pred_depth / gt_depth))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    ##########STEP 2:compute mean depthor###################

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

    return mre, mae, abs_, abs_rel, sq_rel, rmse, rmse_log, log10, a1, a2, a3

def compute_shading_metrics(gt, pred, mask=None, median_align=True):
    """Computation of metrics between predicted and ground truth depths
    """
    gt_depth = gt
    pred_depth = pred

    gt_depth[gt_depth<=0.01] = 0.01
    pred_depth[pred_depth<=0.01] = 0.01
    gt_depth[gt_depth >= 1] = 1
    pred_depth[pred_depth >= 1] = 1


    ###########STEP 1: compute delta#######################
    thresh = torch.max((gt_depth / pred_depth), (pred_depth / gt_depth))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    ##########STEP 2:compute mean depthor###################

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

    return [mre, mae, abs_, abs_rel, sq_rel, rmse, rmse_log, log10, a1, a2, a3]

def compute_alb_metrics(img1, img2, mask):
    """
    Compute Mean Squared Error (MSE) between two RGB images with a mask.

    Args:
    - img1: Tensor, the first image (batch_size x channels x height x width).
    - img2: Tensor, the second image (batch_size x channels x height x width).
    - mask: Tensor, the mask (batch_size x 1 x height x width).

    Returns:
    - mse: Tensor, the MSE loss.
    """
    squared_diff = (img1 - img2)**2
    masked_squared_diff = squared_diff * mask

    # Calculate the mean squared error only for the masked regions
    mse = torch.sum(masked_squared_diff) / torch.sum(mask)
    psnr_ = psnr(img1, img2)
    ssim = ssim_torch(img1, img2)
    return [mse, psnr_, ssim]

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
    return pgp


def mIoU(mask, pred_mask, smooth=1e-10, n_classes=41):
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
        return torch.tensor(np.nanmean(iou_per_class), dtype=torch.float32, device=mask.device)

def pixel_accuracy(mask, output):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        mask = torch.argmax(mask, dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
        accuracy = torch.tensor(accuracy, dtype=torch.float32, device=mask.device)
    return accuracy


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_pred = F.softmax(y_pred, dim=1)
    intersection = torch.sum(y_true * y_pred)
    dice = (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
    return dice

def mean_angular_error(gt,pred):
    gt_n = -1 + 2.0*gt
    pred_n = -1 +2.0*pred
    
    device = torch.device(gt.device)
    
    gt_norm = torch.norm(gt_n, p=2, dim=1, keepdim=True)
    pred_norm = torch.norm(pred_n, p=2, dim=1, keepdim=True)
    pi = 2 * torch.acos(torch.zeros(1,device=device)).item()
    valid_gt_n = gt_norm > 0
    valid_pred_n = pred_norm > 0
    gt_n = gt_n / torch.where(valid_gt_n, gt_norm, torch.tensor(1.0).to(device))
    pred_n = pred_n / torch.where(valid_pred_n, pred_norm, torch.tensor(1.0).to(device))
    # Calculate the cosine of the angular error
    cos_angle_error = torch.clamp(torch.sum(gt_n * pred_n, dim=1), -1.0, 1.0)
    # Calculate the angle in radians
    angle_error_deg  = torch.rad2deg( torch.acos(cos_angle_error) * valid_gt_n )
    pgp5 = torch.mean((angle_error_deg <= 5).float())
    pgp10 = torch.mean((angle_error_deg <= 10).float()) 
    pgp20 = torch.mean((angle_error_deg <= 20).float())
    return angle_error_deg.mean(),pgp5,pgp10,pgp20

def compute_normal_metrics(gt_normal, pred_normal, mask):
    mse, psnr_, ssim = compute_alb_metrics(gt_normal, pred_normal, mask)
    ang, pgp5,pgp10,pgp20 = mean_angular_error(gt_normal, pred_normal)
    return [mse, ang, pgp5, pgp10, pgp20, psnr_, ssim]

def compute_semantic_metrics(gt_semantic, pred_semantic):
    miou = mIoU(gt_semantic, pred_semantic)
    acc = pixel_accuracy(gt_semantic, pred_semantic)
    dice = dice_coefficient(gt_semantic, pred_semantic)
    return [miou, acc, dice]


def psnr(img1, img2, max_val=1.0):
    """
    Compute PSNR between two images.
    
    Parameters:
    img1 (torch.Tensor): First image tensor.
    img2 (torch.Tensor): Second image tensor.
    max_val (float): The maximum possible pixel value of the images.
    
    Returns:
    float: the PSNR value.
    """
    mse = F.mse_loss(img1, img2)
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr

def gaussian_window(size, sigma):
    """
    Generates a 2D Gaussian window.
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords*2) / (2 * sigma*2))
    g /= g.sum()
    return g.outer(g)

def ssim_torch(img1, img2, window_size=11, window_sigma=1.5, size_average=True, val_range=None):
    """
    Computes the SSIM between two images.
    """
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
        
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    window = gaussian_window(window_size, window_sigma).to(img1.device)
    window = window.expand(img1.size(1), 1, window_size, window_size)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
# From https://github.com/fyu/drn
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.vals = []
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.vals.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def to_dict(self):
        return {
            'val': self.val,
            'sum': self.sum,
            'count': self.count,
            'avg': self.avg
        }

    def from_dict(self, meter_dict):
        self.val = meter_dict['val']
        self.sum = meter_dict['sum']
        self.count = meter_dict['count']
        self.avg = meter_dict['avg']


class Evaluator(object):

    def __init__(self, settings, median_align=True):

        self.median_align = median_align
        self.settings = settings
        # depth
        self.metrics = {}
        self.metrics["depth/mre"] = AverageMeter()
        self.metrics["depth/mae"] = AverageMeter()
        self.metrics["depth/abs_"] = AverageMeter()
        self.metrics["depth/abs_rel"] = AverageMeter()
        self.metrics["depth/sq_rel"] = AverageMeter()
        self.metrics["depth/rms"] = AverageMeter()
        self.metrics["depth/log_rms"] = AverageMeter()
        self.metrics["depth/log10"] = AverageMeter()
        self.metrics["depth/a1"] = AverageMeter()
        self.metrics["depth/a2"] = AverageMeter()
        self.metrics["depth/a3"] = AverageMeter()
        ## Shading
        self.metrics["sh/mre"] = AverageMeter()
        self.metrics["sh/mae"] = AverageMeter()
        self.metrics["sh/abs_"] = AverageMeter()
        self.metrics["sh/abs_rel"] = AverageMeter()
        self.metrics["sh/sq_rel"] = AverageMeter()
        self.metrics["sh/rms"] = AverageMeter()
        self.metrics["sh/log_rms"] = AverageMeter()
        self.metrics["sh/log10"] = AverageMeter()
        self.metrics["sh/a1"] = AverageMeter()
        self.metrics["sh/a2"] = AverageMeter()
        self.metrics["sh/a3"] = AverageMeter()
        ##albedo
        self.metrics["alb/mse"] = AverageMeter()
        self.metrics["alb/psnr"] = AverageMeter()
        self.metrics["alb/ssim"] = AverageMeter()
            
        ##Normal
        self.metrics["norm/mse"] = AverageMeter()
        self.metrics["norm/ang_ls"] = AverageMeter()
        self.metrics["norm/pgp5"] = AverageMeter()
        self.metrics["norm/pgp10"] = AverageMeter()
        self.metrics["norm/pgp20"] = AverageMeter()
        self.metrics["norm/psnr"] = AverageMeter()
        self.metrics["norm/ssim"] = AverageMeter()
        ## Semantic
        self.metrics['ss/miou'] = AverageMeter()
        self.metrics['ss/acc'] = AverageMeter()
        self.metrics['ss/dice'] = AverageMeter()

    def reset_eval_metrics(self):
        """
        Resets metrics used to evaluate the model
        """
        # depth
        self.metrics["depth/mre"].reset()
        self.metrics["depth/mae"].reset()
        self.metrics["depth/abs_"].reset()
        self.metrics["depth/abs_rel"].reset()
        self.metrics["depth/sq_rel"].reset()
        self.metrics["depth/rms"].reset()
        self.metrics["depth/log_rms"].reset()
        self.metrics["depth/log10"].reset()
        self.metrics["depth/a1"].reset()
        self.metrics["depth/a2"].reset()
        self.metrics["depth/a3"].reset()
        # shading 
        self.metrics["sh/mre"].reset()
        self.metrics["sh/mae"].reset()
        self.metrics["sh/abs_"].reset()
        self.metrics["sh/abs_rel"].reset()
        self.metrics["sh/sq_rel"].reset()
        self.metrics["sh/rms"].reset()
        self.metrics["sh/log_rms"].reset()
        self.metrics["sh/log10"].reset()
        self.metrics["sh/a1"].reset()
        self.metrics["sh/a2"].reset()
        self.metrics["sh/a3"].reset()
        # albedo 
        self.metrics["alb/mse"].reset()
        self.metrics["alb/psnr"].reset()
        self.metrics["alb/ssim"].reset()
        # Normal 
        self.metrics["norm/mse"].reset()
        self.metrics["norm/ang_ls"].reset()
        self.metrics["norm/pgp5"].reset()
        self.metrics["norm/pgp10"].reset()
        self.metrics["norm/pgp20"].reset()
        self.metrics["norm/psnr"].reset()
        self.metrics["norm/ssim"].reset()
        ## Semantic
        self.metrics['ss/miou'].reset()
        self.metrics['ss/acc'].reset()
        self.metrics['ss/dice'].reset()
        
    def compute_eval_metrics(self, gt_depth=None, pred_depth=None, gt_shading=None, pred_shading=None,
                             gt_albedo=None, pred_albedo=None, gt_normal=None, pred_normal=None, gt_semantic=None,
                             pred_semantic=None,
                             dmask=None, mask = None):
        """
        Computes metrics used to evaluate the model
        """
        N = gt_depth.shape[0]

        mre, mae, abs_, abs_rel, sq_rel, rms, rms_log, log10, a1, a2, a3 = \
            compute_depth_metrics(gt_depth, pred_depth, dmask, self.median_align)

        self.metrics["depth/mre"].update(mre, N)
        self.metrics["depth/mae"].update(mae, N)
        self.metrics["depth/abs_"].update(abs_, N)
        self.metrics["depth/abs_rel"].update(abs_rel, N)
        self.metrics["depth/sq_rel"].update(sq_rel, N)
        self.metrics["depth/rms"].update(rms, N)
        self.metrics["depth/log_rms"].update(rms_log, N)
        self.metrics["depth/log10"].update(log10, N)
        self.metrics["depth/a1"].update(a1, N)
        self.metrics["depth/a2"].update(a2, N)
        self.metrics["depth/a3"].update(a3, N)
        ## shading
        N = gt_shading.shape[0]
        mre, mae, abs_, abs_rel, sq_rel, rms, rms_log, log10, a1, a2, a3 = \
            compute_shading_metrics(gt_shading, pred_shading, dmask, self.median_align)
        self.metrics["sh/mre"].update(mre, N)
        self.metrics["sh/mae"].update(mae, N)
        self.metrics["sh/abs_"].update(abs_, N)
        self.metrics["sh/abs_rel"].update(abs_rel, N)
        self.metrics["sh/sq_rel"].update(sq_rel, N)
        self.metrics["sh/rms"].update(rms, N)
        self.metrics["sh/log_rms"].update(rms_log, N)
        self.metrics["sh/log10"].update(log10, N)
        self.metrics["sh/a1"].update(a1, N)
        self.metrics["sh/a2"].update(a2, N)
        self.metrics["sh/a3"].update(a3, N)
        N = gt_albedo.shape[0]
        # Albedo 
        alb_mse, alb_psnr, alb_ssim = compute_alb_metrics(gt_albedo, pred_albedo, mask)
        self.metrics["alb/mse"].update(alb_mse, N)
        self.metrics["alb/psnr"].update(alb_psnr, N)
        self.metrics["alb/ssim"].update(alb_ssim, N)
        N = gt_normal.shape[0]
        # Normal
        mse, ang_ls, pgp5, pgp10, pgp20,norm_psnr, norm_ssim = compute_normal_metrics(gt_normal, pred_normal, mask) 
        self.metrics["norm/mse"].update(mse, N)
        self.metrics["norm/ang_ls"].update(ang_ls, N)
        self.metrics["norm/pgp5"].update(pgp5, N)
        self.metrics["norm/pgp10"].update(pgp10, N)
        self.metrics["norm/pgp20"].update(pgp20, N)
        self.metrics["norm/psnr"].update(norm_psnr, N)
        self.metrics["norm/ssim"].update(norm_ssim, N)
        N = gt_semantic.shape[0]
        ##Semantic 
        miou, acc, dice = compute_semantic_metrics(gt_semantic*mask, pred_semantic*mask)
        self.metrics['ss/miou'].update(miou, N)
        self.metrics['ss/acc'].update(acc, N)
        self.metrics['ss/dice'].update(dice, N)
        

    def print(self, dir=None):
        avg_metrics = []
        ## depth
        avg_metrics.append(self.metrics["depth/mre"].avg)
        avg_metrics.append(self.metrics["depth/mae"].avg)
        avg_metrics.append(self.metrics["depth/abs_"].avg)
        avg_metrics.append(self.metrics["depth/abs_rel"].avg)
        avg_metrics.append(self.metrics["depth/sq_rel"].avg)
        avg_metrics.append(self.metrics["depth/rms"].avg)
        avg_metrics.append(self.metrics["depth/log_rms"].avg)
        avg_metrics.append(self.metrics["depth/log10"].avg)
        avg_metrics.append(self.metrics["depth/a1"].avg)
        avg_metrics.append(self.metrics["depth/a2"].avg)
        avg_metrics.append(self.metrics["depth/a3"].avg)
        ## Shading
        avg_metrics.append(self.metrics["sh/mre"].avg)
        avg_metrics.append(self.metrics["sh/mae"].avg)
        avg_metrics.append(self.metrics["sh/abs_"].avg)
        avg_metrics.append(self.metrics["sh/abs_rel"].avg)
        avg_metrics.append(self.metrics["sh/sq_rel"].avg)
        avg_metrics.append(self.metrics["sh/rms"].avg)
        avg_metrics.append(self.metrics["sh/log_rms"].avg)
        avg_metrics.append(self.metrics["sh/log10"].avg)
        avg_metrics.append(self.metrics["sh/a1"].avg)
        avg_metrics.append(self.metrics["sh/a2"].avg)
        avg_metrics.append(self.metrics["sh/a3"].avg)
        ##Normal 
        avg_metrics.append(self.metrics["norm/mse"].avg)
        avg_metrics.append(self.metrics["norm/ang_ls"].avg)
        avg_metrics.append(self.metrics["norm/pgp5"].avg)
        avg_metrics.append(self.metrics["norm/pgp10"].avg)
        avg_metrics.append(self.metrics["norm/pgp20"].avg)
        avg_metrics.append(self.metrics["norm/psnr"].avg)
        avg_metrics.append(self.metrics["norm/ssim"].avg)
        ## Semantic
        avg_metrics.append(self.metrics["ss/miou"].avg)
        avg_metrics.append(self.metrics["ss/acc"].avg)
        avg_metrics.append(self.metrics["ss/dice"].avg)
        ##Albeo
        avg_metrics.append(self.metrics["alb/mse"].avg)
        avg_metrics.append(self.metrics["alb/psnr"].avg)
        avg_metrics.append(self.metrics["alb/ssim"].avg)
            
        print("\n********************Depth*******************************")
        print("\n  "+ ("{:>9} | " * 11).format("mre", "mae", "abs_", "abs_rel", "sq_rel", "rms", "rms_log", "log10", "a1", "a2", "a3"))
        print(("&  {: 8.5f} " * 11).format(*avg_metrics[:11]))

        print("\n********************Shading*******************************")
        print("\n  "+ ("{:>9} | " * 11).format("mre", "mae", "abs_", "abs_rel", "sq_rel", "rms", "rms_log", "log10", "a1", "a2", "a3"))
        print(("&  {: 8.5f} " * 11).format(*avg_metrics[11:22]))

        print("\n********************Normal*******************************")
        print("\n  "+ ("{:>9} | " * 7).format("mse", "ang_ls", "pgp5", "pgp10","pgp20","psnr", "ssim"))
        print(("&  {: 8.5f} " * 7).format(*avg_metrics[22:29]))

        print("\n********************Semantic*******************************")
        print("\n  "+ ("{:>9} | " * 3).format("miou", "acc", "dice"))
        print(("&  {: 8.5f} " * 3).format(*avg_metrics[29:32]))

        print("\n********************Albedo*******************************")
        print(" \n" + ("{:>9} | "*3 ).format("mse", "psnr", "ssim"))
        print(("&  {: 8.5f} "*3).format(*avg_metrics[32:]))
        
        if dir is not None:
            file = os.path.join(dir, "result.txt")
            with open(file, 'w') as f:
                print("Depth\n  "+ ("{:>9} | " * 11).format("mre", "mae", "abs_", "abs_rel", "sq_rel", "rms", \
                                                            "rms_log", "log10", "a1", "a2", "a3"), file = f)
                print(("&  {: 8.5f} " * 11).format(*avg_metrics[:11]), file = f)
        
                print("shading\n  "+ ("{:>9} | " * 11).format("mre", "mae", "abs_", "abs_rel", "sq_rel", "rms", \
                                                              "rms_log", "log10", "a1", "a2", "a3"), file = f)
                print(("&  {: 8.5f} " * 11).format(*avg_metrics[11:22]), file = f)
                
                print("normal\n  "+ ("{:>9} | " * 5).format("mse", "ang_ls", "pgp", "psnr", "ssim"), file = f)
                print(("&  {: 8.5f} " * 5).format(*avg_metrics[22:27]), file = f)
                
                # print("semantic\n  "+ ("{:>9} | " * 3).format("miou", "acc", "dice"))
                # print(("&  {: 8.5f} " * 3).format(*avg_metrics[27:30]), file = f)
        
                print("albedo \n" + ("{:>9} | "*3 ).format("mse"), file = f)
                print(("&  {: 8.5f} "*3).format(avg_metrics[30:]))
