import SimpleITK as sitk
import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize,skeletonize_3d
from scipy import optimize
from utils.transform_matrix import w_to_T
import os
import time

from .proj_weight import get_proj_weights, get_eigen_list, update_proj_weights
from .topo_weight import get_topo_weights,create_graph,remove_noise_points,obtain_branches,get_point_indices_from_branches

#世界坐标系：以放射源到接收器中心的连线和patient plane之间的交点为原点，垂直向上为z轴正方向，LAO为x轴正方向，CRA为y轴正方向
#相机坐标系：放射源为原点，放射源到图像中心的射线方向为z轴正方向，x轴y轴方向与像素坐标系的x轴y轴方向保持一致
#图像坐标系：图像中心为原点，向右为x轴正方向，向下为y轴正方向
#像素坐标系：图像左上角为原点，向右为x轴正方向，向下为y轴正方向


def distance(w:np.ndarray,P:np.ndarray,x:np.ndarray,y:np.ndarray):

    num_points = np.min([x.shape[1],y.shape[1]])
    x_optim = x[:,0:num_points]
    y_optim = y[:,0:num_points]

    T = w_to_T(w)

    x_proj = func(P,T,x_optim)
    mean_distance = np.mean(np.sqrt((y_optim[0]-x_proj[0])**2+(y_optim[1]-x_proj[1])**2))

    return mean_distance


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


def adaptive_error(w:np.ndarray,
                   P:np.ndarray,
                   x:np.ndarray,
                   y:np.ndarray,
                   weights:np.ndarray):

    T = w_to_T(w)

    x_proj = func(P,T,x)
    err = np.sum(weights*((y[0]-x_proj[0])**2+(y[1]-x_proj[1])**2))

    return err


def adaptive_error_two_view(w:np.ndarray,
                            x:np.ndarray,
                            P_cor:np.ndarray,
                            y_cor:np.ndarray,
                            weights_cor:np.ndarray,
                            P_sag:np.ndarray,
                            y_sag:np.ndarray,
                            weights_sag:np.ndarray):

    err_cor = adaptive_error(w,P_cor,x,y_cor,weights_cor)
    err_sag = adaptive_error(w,P_sag,x,y_sag,weights_sag)
    # err_cor =0
    # err_sag = 0

    return err_cor+err_sag

def ICP_PWO(array_3d,
                                        array_2d_cor, 
                                        array_2d_sag,
                                        T_3d_voxel_to_img,
                                        w_0,
                                        P_cor,
                                        angles_cor,
                                        P_sag,
                                        angles_sag,
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

    """计算三维血管中心线各点的拓扑权重"""
    _,_,topo_weights = get_topo_weights(s1_0)

    """计算三维血管中心线在正侧两个平面上的投影权重"""
    #获取各点处的最佳投影平面
    eigen_values_list, eigen_vectors_list = get_eigen_list(s1_0,50)

    #正位投影方向
    angle1_cor = angles_cor[0]
    angle2_cor = angles_cor[1]
    proj_vector_cor = np.array([np.cos(angle2_cor)*np.sin(angle1_cor),
                                np.sin(angle2_cor),
                                np.cos(angle2_cor)*np.cos(angle1_cor)])


    #侧位投影方向
    angle1_sag = angles_sag[0]
    angle2_sag = angles_sag[1]
    proj_vector_sag = np.array([np.cos(angle2_sag)*np.sin(angle1_sag),
                                np.sin(angle2_sag),
                                np.cos(angle2_sag)*np.cos(angle1_sag)])


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

    """显示1/3"""
    # fig = plt.figure()
    # ax_3d_cor = fig.add_subplot(221,projection="3d")
    # ax_cor = fig.add_subplot(222)
    # ax_3d_sag = fig.add_subplot(223,projection="3d")
    # ax_sag = fig.add_subplot(224)


    """迭代优化"""
    w_old = w_0
    w = w_0
    T = w_to_T(w_0)
    i = 0
    while(1):

        """根据T对血管中心线的原始三维坐标进行调整"""
        s1 = T@s1_0

        """更新权重"""
        #更新正位权重
        proj_weights_cor = update_proj_weights(eigen_values_list,
                                               eigen_vectors_list,
                                               T,
                                               proj_vector_cor)
        weights_cor = topo_weights*proj_weights_cor
        # weights_cor = proj_weights_cor

        #更新侧位权重
        proj_weights_sag = update_proj_weights(eigen_values_list,
                                               eigen_vectors_list,
                                               T,
                                               proj_vector_sag)
        weights_sag = topo_weights*proj_weights_sag
        # weights_sag = proj_weights_sag



        """计算相似度,并进行点匹配"""
        s1_p_cor = P_cor@s1
        s1_p_cor = s1_p_cor/np.expand_dims(s1_p_cor[2],axis=0)
        near_point_id_cor = similarity(s1_p_cor[0:2],s2_cor[0:2])
        near_point_cor = s2_cor[:,near_point_id_cor]
        
        s1_p_sag = P_sag@s1
        s1_p_sag = s1_p_sag/np.expand_dims(s1_p_sag[2],axis=0)
        near_point_id_sag = similarity(s1_p_sag[0:2],s2_sag[0:2])
        near_point_sag = s2_sag[:,near_point_id_sag]


        """显示2/3"""
        # ax_3d_cor.cla()
        # ax_cor.cla()
        # ax_3d_sag.cla()
        # ax_sag.cla()
        # ax_min = np.floor(s1[0:3].min(axis = 1)/10)*10
        # ax_max = np.ceil(s1[0:3].max(axis = 1)/10)*10
        
        # #正位
        # ax_3d_cor.set_xticks(np.linspace(ax_min[0], ax_max[0], int((ax_max[0]-ax_min[0])/5))+1)
        # ax_3d_cor.set_yticks(np.linspace(ax_min[1], ax_max[1], int((ax_max[1]-ax_min[1])/5))+1)
        # ax_3d_cor.set_zticks(np.linspace(ax_min[2], ax_max[2], int((ax_max[2]-ax_min[0])/5))+1)
        # ax_3d_cor.set_box_aspect([ax_max[0] - ax_min[0],ax_max[1] - ax_min[1],ax_max[2] - ax_min[2]])

        # ax_3d_cor.scatter(s1[0],s1[1],s1[2],c=proj_weights_cor)
        # ax_3d_cor.quiver(0,
        #           0,
        #           0,
        #           20*proj_vector_cor[0],
        #           20*proj_vector_cor[1],
        #           20*proj_vector_cor[2])
        
        # ax_cor.scatter(s2_cor[0],s2_cor[1],s2_cor[2])
        # ax_cor.scatter(s1_p_cor[0],s1_p_cor[1],s1_p_cor[2],c=proj_weights_cor)
        
        # #侧位
        # ax_3d_sag.set_xticks(np.linspace(ax_min[0], ax_max[0], int((ax_max[0]-ax_min[0])/5))+1)
        # ax_3d_sag.set_yticks(np.linspace(ax_min[1], ax_max[1], int((ax_max[1]-ax_min[1])/5))+1)
        # ax_3d_sag.set_zticks(np.linspace(ax_min[2], ax_max[2], int((ax_max[2]-ax_min[0])/5))+1)
        # ax_3d_sag.set_box_aspect([ax_max[0] - ax_min[0],ax_max[1] - ax_min[1],ax_max[2] - ax_min[2]])

        # ax_3d_sag.scatter(s1[0],s1[1],s1[2],c=proj_weights_sag)
        # ax_3d_sag.quiver(0,
        #           0,
        #           0,
        #           20*proj_vector_sag[0],
        #           20*proj_vector_sag[1],
        #           20*proj_vector_sag[2])
        # ax_sag.scatter(s2_sag[0],s2_sag[1],s2_sag[2])
        # ax_sag.scatter(s1_p_sag[0],s1_p_sag[1],s1_p_sag[2],c=proj_weights_sag)
        # plt.show()

        """进行优化更新T"""
        problem = optimize.OptimizeResult(
            fun = adaptive_error_two_view,
            x0 = w,
            args = (s1_0,
                    P_cor,near_point_cor,weights_cor,
                    P_sag,near_point_sag,weights_sag),
            # bounds = [(-10000,10000),(-10000,10000),(-10000,10000),
            #           (-500,500),(-500,500),(-500,500)]
        )
        optim_result = optimize.minimize(**problem)
        w = optim_result.x
        T = w_to_T(w)


        obj_err = optim_result.fun
        obj_err_cor = distance(w,P_cor,s1_0,near_point_cor)/s1_0.shape[1]
        obj_err_sag = distance(w,P_sag,s1_0,near_point_sag)/s1_0.shape[1]
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


