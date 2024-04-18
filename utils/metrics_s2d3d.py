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


def mIoU(mask, pred_mask, smooth=1e-10, n_classes=14):
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

def compute_semantic_metrics(gt_semantic, pred_semantic):
    miou = mIoU(gt_semantic, pred_semantic)
    acc = pixel_accuracy(gt_semantic, pred_semantic)
    dice = dice_coefficient(gt_semantic, pred_semantic)
    return [miou, acc, dice]


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
        # depthor and Accuracy metric trackers
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
        self.metrics['ss/miou'].reset()
        self.metrics['ss/acc'].reset()
        self.metrics['ss/dice'].reset()
        
    def compute_eval_metrics(self, gt_depth=None, pred_depth=None, gt_semantic=None,
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
        N = gt_semantic.shape[0]
        ##Semantic 
        miou, acc, dice = compute_semantic_metrics(gt_semantic*mask, pred_semantic*mask)
        self.metrics['ss/miou'].update(miou, N)
        self.metrics['ss/acc'].update(acc, N)
        self.metrics['ss/dice'].update(dice, N)
        

    def print(self, dir=None):
        avg_metrics = []
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
        
        avg_metrics.append(self.metrics["ss/miou"].avg)
        avg_metrics.append(self.metrics["ss/acc"].avg)
        avg_metrics.append(self.metrics["ss/dice"].avg)
        
        print("\n********************Depth*******************************")
        print("\n  "+ ("{:>9} | " * 11).format("mre", "mae", "abs_", "abs_rel", "sq_rel", "rms", "rms_log", "log10", "a1", "a2", "a3"))
        print(("&  {: 8.5f} " * 11).format(*avg_metrics[:11]))

        print("\n********************Semantic*******************************")
        print("\n  "+ ("{:>9} | " * 3).format("miou", "acc", "dice"))
        print(("&  {: 8.5f} " * 3).format(*avg_metrics[11:]))

        if dir is not None:
            file = os.path.join(dir, "result.txt")
            with open(file, 'w') as f:
                print("Depth\n  "+ ("{:>9} | " * 11).format("mre", "mae", "abs_", "abs_rel", "sq_rel", "rms", \
                                                            "rms_log", "log10", "a1", "a2", "a3"), file = f)
                print(("&  {: 8.5f} " * 11).format(*avg_metrics[:11]), file = f)
    
                print("semantic\n  "+ ("{:>9} | " * 3).format("miou", "acc", "dice"))
                print(("&  {: 8.5f} " * 3).format(*avg_metrics[11:]), file = f)
        
