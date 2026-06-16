import SimpleITK as sitk
import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize,skeletonize_3d
from scipy import optimize
from utils.transform_matrix import w_to_T
import os
import time



#世界坐标系：以放射源到接收器中心的连线和patient plane之间的交点为原点，垂直向上为z轴正方向，LAO为x轴正方向，CRA为y轴正方向
#相机坐标系：放射源为原点，放射源到图像中心的射线方向为z轴正方向，x轴y轴方向与像素坐标系的x轴y轴方向保持一致
#图像坐标系：图像中心为原点，向右为x轴正方向，向下为y轴正方向
#像素坐标系：图像左上角为原点，向右为x轴正方向，向下为y轴正方向



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



"""优化问题准备"""
def func(P:np.ndarray,T:np.ndarray,x:np.ndarray):
    """
    计算三维点x投影后的像素坐标
    Input:
    P-[3,4]
    T-[3,4]
    x-[4,N]
    Return:
    x_proj-[3,N]
    """

    x_proj = P@T@x
    x_proj = x_proj/np.expand_dims(x_proj[2],axis=0)

    return x_proj


def error(w:np.ndarray,P:np.ndarray,x:np.ndarray,y:np.ndarray):

    T = w_to_T(w)

    x_proj = func(P,T,x)
    err = np.sum((y[0]-x_proj[0])**2+(y[1]-x_proj[1])**2)

    return err


def error_two_view(w:np.ndarray,x:np.ndarray,P_cor:np.ndarray,y_cor:np.ndarray,P_sag:np.ndarray,y_sag:np.ndarray):

    err_cor = error(w,P_cor,x,y_cor)
    err_sag = error(w,P_sag,x,y_sag)
    # err_cor =0
    # err_sag = 0

    return err_cor+err_sag


def ICP_for_2d_3d_registration(array_3d,
                               array_2d_cor, 
                               array_2d_sag,
                               T_3d_voxel_to_img,
                               w_0,
                               P_cor,
                               P_sag,
                               err_limit=0.1):
    """
    Input: 
    array_3d-[H,W,D]
    array_2d_cor-[H,W]
    array_2d_sag-[H,W]w_0
    Output: 
    T-[4,4],三维数据的坐标变换矩阵
    """

    """获取血管中心线的三维坐标"""
    #细化三维血管
    skeleton_3d = skeletonize_3d(array_3d)
    #获取中心线三维坐标
    skeleton_3d_xyz = np.argwhere(skeleton_3d>0)#[N,3]
    s1_0 = np.column_stack((skeleton_3d_xyz,np.ones((len(skeleton_3d_xyz),1)))).T #[4,N]
    #将中心线三维坐标
    s1_0 = T_3d_voxel_to_img@s1_0

    """获取血管中心线在coronal平面的像素坐标"""
    #细化二维血管
    skeleton_2d_cor = skeletonize(array_2d_cor)
    #获取中心线二维坐标
    skeleton_2d_cor_xyz = np.argwhere(skeleton_2d_cor>0)
    s2_cor = np.column_stack((skeleton_2d_cor_xyz,np.ones((len(skeleton_2d_cor_xyz),1)))).T #[3,N]

    """获取血管中心线在saggital平面的像素坐标"""
    #细化二维血管
    skeleton_2d_sag = skeletonize(array_2d_sag)
    #获取中心线二维坐标
    skeleton_2d_sag_xyz = np.argwhere(skeleton_2d_sag>0)
    s2_sag = np.column_stack((skeleton_2d_sag_xyz,np.ones((len(skeleton_2d_sag_xyz),1)))).T #[3,N]

    """迭代优化"""
    w_old = w_0
    w = w_0
    T = w_to_T(w_0)
    i = 0
    while(1):

        """根据T对血管中心线的原始三维坐标进行调整"""
        s1 = T@s1_0

        """计算s1的权重[N]"""


        """计算相似度,并进行点匹配"""
        s1_p_cor = P_cor@s1
        s1_p_cor = s1_p_cor/np.expand_dims(s1_p_cor[2],axis=0)
        near_point_id_cor = similarity(s1_p_cor[0:2],s2_cor[0:2])
        near_point_cor = s2_cor[:,near_point_id_cor]
        
        s1_p_sag = P_sag@s1
        s1_p_sag = s1_p_sag/np.expand_dims(s1_p_sag[2],axis=0)
        near_point_id_sag = similarity(s1_p_sag[0:2],s2_sag[0:2])
        near_point_sag = s2_sag[:,near_point_id_sag]

        """进行优化更新T"""
        problem = optimize.OptimizeResult(
            fun = error_two_view,
            x0 = w,
            args = (s1_0,P_cor,near_point_cor,P_sag,near_point_sag),
            # bounds = [(-10000,10000),(-10000,10000),(-10000,10000),
            #           (-500,500),(-500,500),(-500,500)]
        )
        optim_result = optimize.minimize(**problem)
        w = optim_result.x
        T = w_to_T(w)


        obj_err = optim_result.fun
        obj_err_cor = error(w,P_cor,s1_0,near_point_cor)/s1_0.shape[1]
        obj_err_sag = error(w,P_sag,s1_0,near_point_sag)/s1_0.shape[1]
        print("Epoch {} error:{}".format(i,obj_err))

        """计算前一次位姿和当前位姿的偏移量,过小则终止循环"""
        err = np.sum((np.array(w)-np.array(w_old))**2)        
        if err < err_limit:
            break

        """更新"""
        w_old = w
        i = i+1
    
    print("registration done!")
    return w,T,obj_err_cor,obj_err_sag





