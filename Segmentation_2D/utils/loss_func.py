import torch
import torch.nn.functional as F
from monai.losses.dice import DiceLoss, DiceCELoss, DiceFocalLoss



def mean_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor
    )->torch.Tensor:
    """
    Params
    ------
    pred: Tensor(B, C, N)
    target: Tensor(B, 1, N)
    
    Return
    ------
    loss: Tensor
    """

    dice_loss = DiceLoss(to_onehot_y=True,softmax=True)

    return dice_loss(pred,target)

def dice_ce_loss(
    pred: torch.Tensor,
    target: torch.Tensor
    )->torch.Tensor:
    """
    Params
    ------
    pred: Tensor(B, C, N)
    target: Tensor(B, 1, N)
    
    Return
    ------
    loss: Tensor
    """

    dice_loss = DiceCELoss(to_onehot_y=True,softmax=True)

    return dice_loss(pred,target)

def dice_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor
    )->torch.Tensor:
    """
    Params
    ------
    pred: Tensor(B, C, N)
    target: Tensor(B, 1, N)
    
    Return
    ------
    loss: Tensor
    """

    dice_loss = DiceFocalLoss(to_onehot_y=True,softmax=True)

    return dice_loss(pred,target)


