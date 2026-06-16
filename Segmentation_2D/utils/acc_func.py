import torch
from monai.networks.utils import one_hot
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from monai.metrics import MeanIoU

from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.metrics import roc_auc_score
import numpy as np
from skimage.morphology import skeletonize, skeletonize_3d


def accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    )->torch.Tensor:
    """
    Params
    ------
    pred: Tensor(B, C, D, W, H)
    target: Tensor(B, 1, D, W, H)
    if_discrete: 需要将标签值硬化
    
    Return
    ------
    dice: Tensor(C) 该批次内各个通道的平均dice值
    """

    #对pred二值化
    num_class = pred.shape[1]
    pred = pred.transpose(1,0)#(C, B, D, W, H) 
    bin_transform = AsDiscrete(argmax=True,to_onehot=num_class)
    pred = bin_transform(pred).as_tensor() #(C, B, D, W, H)
    pred = pred.transpose(1,0)#(B, C, D, W, H)

    #对target进行one-hot编码
    target = one_hot(target,num_class)#(B, C, D, W, H)

    #计算该批次内通道的指标值
    pred_flatten = pred.permute(1,0,2,3).flatten(start_dim=1).numpy()
    target_flatten = target.permute(1,0,2,3).flatten(start_dim=1).numpy()
    
    values = []
    for i in range(num_class):
        value = accuracy_score(target_flatten[i],pred_flatten[i])
        values.append(value)

    values = torch.tensor(values)
    return values


def precision(
    pred: torch.Tensor,
    target: torch.Tensor,
    )->torch.Tensor:
    """
    Params
    ------
    pred: Tensor(B, C, D, W, H)
    target: Tensor(B, 1, D, W, H)
    if_discrete: 需要将标签值硬化
    
    Return
    ------
    dice: Tensor(C) 该批次内各个通道的平均dice值
    """

    #对pred二值化
    num_class = pred.shape[1]
    pred = pred.transpose(1,0)#(C, B, D, W, H) 
    bin_transform = AsDiscrete(argmax=True,to_onehot=num_class)
    pred = bin_transform(pred).as_tensor() #(C, B, D, W, H)
    pred = pred.transpose(1,0)#(B, C, D, W, H)

    #对target进行one-hot编码
    target = one_hot(target,num_class)#(B, C, D, W, H)

    #计算该批次内通道的指标值
    pred_flatten = pred.permute(1,0,2,3).flatten(start_dim=1).numpy()
    target_flatten = target.permute(1,0,2,3).flatten(start_dim=1).numpy()
    
    values = []
    for i in range(num_class):
        value = precision_score(target_flatten[i],pred_flatten[i])
        values.append(value)

    values = torch.tensor(values)
    return values


def sensitivity(
    pred: torch.Tensor,
    target: torch.Tensor,
    )->torch.Tensor:
    """
    Params
    ------
    pred: Tensor(B, C, D, W, H)
    target: Tensor(B, 1, D, W, H)
    if_discrete: 需要将标签值硬化
    
    Return
    ------
    dice: Tensor(C) 该批次内各个通道的平均dice值
    """

    #对pred二值化
    num_class = pred.shape[1]
    pred = pred.transpose(1,0)#(C, B, D, W, H) 
    bin_transform = AsDiscrete(argmax=True,to_onehot=num_class)
    pred = bin_transform(pred).as_tensor() #(C, B, D, W, H)
    pred = pred.transpose(1,0)#(B, C, D, W, H)

    #对target进行one-hot编码
    target = one_hot(target,num_class)#(B, C, D, W, H)

    #计算该批次内通道的指标值
    pred_flatten = pred.permute(1,0,2,3).flatten(start_dim=1).numpy()
    target_flatten = target.permute(1,0,2,3).flatten(start_dim=1).numpy()
    
    values = []
    for i in range(num_class):
        value = recall_score(target_flatten[i],pred_flatten[i])
        values.append(value)

    values = torch.tensor(values)
    return values

def roc_auc(
    pred: torch.Tensor,
    target: torch.Tensor,
    )->torch.Tensor:
    """
    Params
    ------
    pred: Tensor(B, C, D, W, H)
    target: Tensor(B, 1, D, W, H)
    if_discrete: 需要将标签值硬化
    
    Return
    ------
    dice: Tensor(C) 该批次内各个通道的平均dice值
    """

    #对pred二值化
    num_class = pred.shape[1]
    pred = pred.transpose(1,0)#(C, B, D, W, H) 
    bin_transform = AsDiscrete(argmax=True,to_onehot=num_class)
    pred = bin_transform(pred).as_tensor() #(C, B, D, W, H)
    pred = pred.transpose(1,0)#(B, C, D, W, H)

    #对target进行one-hot编码
    target = one_hot(target,num_class)#(B, C, D, W, H)

    #计算该批次内通道的指标值
    pred_flatten = pred.permute(1,0,2,3).flatten(start_dim=1).numpy()
    target_flatten = target.permute(1,0,2,3).flatten(start_dim=1).numpy()
    
    values = []
    for i in range(num_class):
        value = roc_auc_score(target_flatten[i],pred_flatten[i])
        values.append(value)

    values = torch.tensor(values)
    return values

def dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    )->torch.Tensor:
    """
    Params
    ------
    pred: Tensor(B, C, D, W, H)
    target: Tensor(B, 1, D, W, H)
    if_discrete: 需要将标签值硬化
    
    Return
    ------
    dice: Tensor(C) 该批次内各个通道的平均dice值
    """

    #对pred二值化
    num_class = pred.shape[1]
    pred = pred.transpose(1,0)#(C, B, D, W, H) 
    bin_transform = AsDiscrete(argmax=True,to_onehot=num_class)
    pred = bin_transform(pred).as_tensor() #(C, B, D, W, H)
    pred = pred.transpose(1,0)#(B, C, D, W, H)

    #对target进行one-hot编码
    target = one_hot(target,num_class)#(B, C, D, W, H)

    #计算该批次内通道的dice值
    dice_metric = DiceMetric(reduction="mean_batch")
    dice_metric(pred, target)
    dice = dice_metric.aggregate()
    dice_metric.reset()

    return dice


def mIoU(
    pred:torch.Tensor,
    target:torch.Tensor
    )->torch.Tensor:
    """
    Params
    ------
    pred: Tensor(B, C, D, W, H)
    target: Tensor(B, 1, D, W, H)
    
    Return
    ------
    dice: Tensor(C) 该批次内各个通道的平均dice值
    """

    #对pred二值化
    num_class = pred.shape[1]
    pred = pred.transpose(1,0)#(C, B, D, W, H)
    bin_transform = AsDiscrete(argmax=True,to_onehot=num_class)
    pred = bin_transform(pred).as_tensor() #(C, B, D, W, H)
    pred = pred.transpose(1,0)#(B, C, D, W, H)

    #对target进行one-hot编码
    target = one_hot(target,num_class)#(B, C, D, W, H)

    #计算该批次内各个通道的hausdorff distance
    mIoU_metric = MeanIoU(reduction="mean_batch")
    mIoU_metric(pred,target)
    mIoU = mIoU_metric.aggregate()
    mIoU_metric.reset

    return mIoU

def hd95(
    pred:torch.Tensor,
    target:torch.Tensor
    )->torch.Tensor:
    """
    Params
    ------
    pred: Tensor(B, C, D, W, H)
    target: Tensor(B, 1, D, W, H)
    
    Return
    ------
    dice: Tensor(C) 该批次内各个通道的平均dice值
    """

    #对pred二值化
    num_class = pred.shape[1]
    pred = pred.transpose(1,0)#(C, B, D, W, H)
    bin_transform = AsDiscrete(argmax=True,to_onehot=num_class)
    pred = bin_transform(pred).as_tensor() #(C, B, D, W, H)
    pred = pred.transpose(1,0)#(B, C, D, W, H)

    #对target进行one-hot编码
    target = one_hot(target,num_class)#(B, C, D, W, H)

    #计算该批次内各个通道的hausdorff distance
    hd_metric = HausdorffDistanceMetric(include_background=True,percentile=95,reduction="mean_batch")
    hd_metric(pred,target)
    hd = hd_metric.aggregate()
    hd_metric.reset

    return hd



"""clDice"""
def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(pred:torch.Tensor, 
           target:torch.Tensor):
    """[this function computes the cldice metric]

    Args:
        pred: Tensor(B, C, D, W, H)
        target: Tensor(B, 1, D, W, H)

    Returns:
        [float]: [cldice metric]
    """
    
    num_class = pred.shape[1]
    target = one_hot(target,num_class)#(B, C, D, W, H)
    
    pred = pred.numpy()
    target = target.numpy()
    
    clDice_class = [torch.tensor(0)]
    
    for c in range(1,num_class):
    
        if len(pred.shape)==4:
            
            v_p = pred[:,c,:,:]
            v_l = target[:,c,:,:]
            
            clDice_batch = []
            for b in range(pred.shape[0]):
                tprec = cl_score(v_p[b],skeletonize(v_l[b]))
                tsens = cl_score(v_l[b],skeletonize(v_p[b]))
                clDice = 2*tprec*tsens/(tprec+tsens)
                clDice_batch.append(clDice)
            
            clDice_batch = np.array(clDice_batch)
            clDice_batch_mean = torch.tensor(clDice_batch.mean())
            
            clDice_class.append(clDice_batch_mean)
                
            

        elif len(pred.shape)==5:
            
            v_p = pred[:,c,:,:,:]
            v_l = target[:,c,:,:,:]
            
            clDice_batch = []
            for b in range(pred.shape[0]):
                tprec = cl_score(v_p[b],skeletonize_3d(v_l[b]))
                tsens = cl_score(v_l[b],skeletonize_3d(v_p[b]))
                clDice = 2*tprec*tsens/(tprec+tsens)
                clDice_batch.append(clDice)
            
            clDice_batch = np.array(clDice_batch)
            clDice_batch_mean = torch.tensor(clDice_batch.mean())
            
            clDice_class.append(clDice_batch_mean)
        
    
    return clDice_class


