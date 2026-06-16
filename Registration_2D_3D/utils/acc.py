import torch
import numpy as np
from monai.metrics import DiceMetric
from skimage.morphology import skeletonize,skeletonize_3d

from .transform_matrix import w_to_T


def dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    )->torch.Tensor:
    """
    Params
    ------
    pred: Tensor(B, 1, D, W, H)
    target: Tensor(B, 1, D, W, H)
    if_discrete: 需要将标签值硬化
    
    Return
    ------
    dice: Tensor(C) 该批次内各个通道的平均dice值
    """


    #计算该批次内通道的dice值
    dice_metric = DiceMetric(reduction="mean_batch")
    dice_metric(pred, target)
    dice = dice_metric.aggregate()
    dice_metric.reset()

    return dice


"""计算两个二维点集之间的相似度"""
def similarity(p_source, p_target):
    """
    计算p_souce各点到p_target各点的最短距离
    Input:
    p_source-[2,N1]
    p_target-[2,N2]
    Return:
    distance-[N1]
    """ 

    #p_source-[N1,2],p_target-[N2,2]
    p_source = p_source.T
    p_target = p_target.T

    min_ids = []
    distance_mask = np.zeros(p_target.shape[0])
    count = np.zeros(p_target.shape[0])

    for p in p_source:
        distance = np.sum((np.expand_dims(p,axis=0)-p_target)**2,axis=-1)
        distance = distance+distance_mask
        min_id = np.argmin(distance)
        min_ids.append(min_id)

        # count[min_id] = count[min_id]+1
        # if count[min_id] > 1:
        #     distance_mask[min_id] = 10000000

    return min_ids


"""计算平均距离误差"""
def mean_skeleton_distance(array_3d,
                           T_3d_voxel_to_img,
                           w,
                           array_2d,
                           P):
    """获取血管中心线的三维坐标"""
    #细化三维血管
    skeleton_3d = skeletonize_3d(array_3d)
    #获取中心线三维坐标
    skeleton_3d_xyz = np.argwhere(skeleton_3d>0)#[N,3]
    s1_0 = np.column_stack((skeleton_3d_xyz,np.ones((len(skeleton_3d_xyz),1)))).T #[4,N]
    #将中心线三维坐标
    s1_0 = T_3d_voxel_to_img@s1_0
    T = w_to_T(w)
    s1 = T@s1_0

    """获取血管中心线在coronal平面的像素坐标"""
    #细化二维血管
    skeleton_2d = skeletonize(array_2d)
    #获取中心线二维坐标
    skeleton_2d_xyz = np.argwhere(skeleton_2d>0)
    s2 = np.column_stack((skeleton_2d_xyz,np.ones((len(skeleton_2d_xyz),1)))).T #[3,N]



    """计算相似度,并进行点匹配"""
    s1_p = P@s1
    s1_p = s1_p/np.expand_dims(s1_p[2],axis=0)
    near_point_id = similarity(s1_p[0:2],s2[0:2])
    near_point = s2[:,near_point_id]


    err1 = np.mean(np.sqrt((near_point[0]-s1_p[0])**2+(near_point[1]-s1_p[1])**2))

    # err1 = np.sum((near_point[0]-s1_0_moving[0])**2+(near_point[1]-s1_0_moving[1])**2+(near_point[2]-s1_0_moving[2])**2)/near_point.shape[1]

    return err1

"""计算平均距离误差"""
def mean_projection_distance(s1_0,
                           T_3d_voxel_to_img,
                           w_target,
                           w_pred,
                           P):
    
    """将三维血管中心线的体素坐标变为世界坐标系"""
    s1_0 = T_3d_voxel_to_img@s1_0

    """目标的点云"""
    T_target = w_to_T(w_target)
    s1_target = T_target@s1_0

    s1_target_p = P@s1_target
    s1_target_p = s1_target_p/np.expand_dims(s1_target_p[2],axis=0)

    """真实点云"""
    T_pred = w_to_T(w_pred)
    s1_pred = T_pred@s1_0

    s1_pred_p = P@s1_pred
    s1_pred_p = s1_pred_p/np.expand_dims(s1_pred_p[2],axis=0)


    err1 = np.mean(np.sqrt((s1_pred_p[0]-s1_target_p[0])**2 + (s1_pred_p[1]-s1_target_p[1])**2))


    return err1


"""计算平均距离误差"""
def mean_target_registration_distance(s1_0,
                           T_3d_voxel_to_img,
                           w_target,
                           w_pred,
                           ):
    
    """将三维血管中心线的体素坐标变为世界坐标系"""
    s1_0 = T_3d_voxel_to_img@s1_0

    """目标的点云"""
    T_target = w_to_T(w_target)
    s1_target = T_target@s1_0


    """真实点云"""
    T_pred = w_to_T(w_pred)
    s1_pred = T_pred@s1_0



    err1 = np.mean(np.sqrt((s1_pred[0]-s1_target[0])**2 + (s1_pred[1]-s1_target[1])**2))


    return err1