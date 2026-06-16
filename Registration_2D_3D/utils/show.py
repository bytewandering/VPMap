from skimage.morphology import skeletonize,skeletonize_3d
import numpy as np
import matplotlib.pyplot as plt

from methods.ICB.topo_weight import get_topo_weights,create_graph,remove_noise_points,obtain_branches,get_point_indices_from_branches
from methods.ICB.icb import get_branch_points
from utils.transform_matrix import w_to_T


def show(P,
         T_3d_voxel_to_img,
         w,
         array_2d,
         skele_3d_points_voxel,
         skele_2d_points_pixel,
         save_path = None):
    
    T = w_to_T(w)

    """获取血管中心线的三维坐标"""
    # #细化三维血管
    # skele_3d_volume = skeletonize_3d(array_3d)
    # #获取中心线三维坐标(体素坐标)
    # skele_3d_points_voxel = np.argwhere(skele_3d_volume>0)#[N,3]
    # skele_3d_points_voxel = np.column_stack((skele_3d_points_voxel,np.ones((len(skele_3d_points_voxel),1)))).T #[4,N]
    #获取中心线三维坐标(图像坐标系)
    skele_3d_points_volume = T@T_3d_voxel_to_img@skele_3d_points_voxel




    """获取血管中心线在coronal平面的像素坐标"""
    # #细化二维血管
    # skele_2d_image = skeletonize(array_2d)
    # #获取中心线二维坐标
    # skele_2d_points_pixel = np.argwhere(skele_2d_image>0)[:,0:2]
    # skele_2d_points_pixel = np.column_stack((skele_2d_points_pixel,np.ones((len(skele_2d_points_pixel),1)))).T #[3,N]

    s1_p = P@skele_3d_points_volume
    s1_p = s1_p/np.expand_dims(s1_p[2],axis=0)



    """显示1/3"""
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)



    ax.set_xlim(0,array_2d.shape[0])
    ax.set_ylim(array_2d.shape[1],0)
    ax.set_box_aspect(1)



    ax.scatter(skele_2d_points_pixel[0],
               skele_2d_points_pixel[1],
               s=1)


    ax.scatter(s1_p[0],
               s1_p[1],
               s=1)

    
    if save_path is not None:
        plt.savefig(save_path,bbox_inches='tight')
    plt.close()


    