import SimpleITK as sitk
import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.morphology import skeletonize_3d
from sklearn.decomposition import PCA
from utils.read import read_dsa_2d_ori
from utils.transform_matrix import w_to_T

import os
import time


def get_eigen_list(skele_points:np.ndarray, num_adj_points:int):

    """
    Input:
    skele_points-血管的三维中心线点
    graph-中心线点组成的拓扑图
    num_adj_points-用于计算最佳投影角度的近邻点个数
    proj_vector-当前DSA机器的投影方向
    """
    N = skele_points.shape[1]
    eigen_values_list = []
    eigen_vectors_list = []
    
    for i in range(N):
        """对每个点取其近邻点num_adj_points个"""
        point = np.expand_dims(skele_points[0:3,i],axis=1)
        dists = np.sum((skele_points[0:3]-point)**2,axis=0)


        adj_point_indices = np.argsort(dists)[0:num_adj_points]

        #adj_points-[3,num_adj_points]
        adj_points = skele_points[0:3,adj_point_indices.tolist()]

        """计算近邻点集的最佳投影方向"""
        pca = PCA(n_components=2)
        pca.fit(adj_points.T)

        eigenvalues = pca.explained_variance_
        eigen_values_list.append(eigenvalues)

        eigenvectors = pca.components_
        eigen_vectors_list.append(eigenvectors)

    eigen_values_array = np.array(eigen_values_list)
    eigen_vectors_array = np.array(eigen_vectors_list)
    return eigen_values_array,eigen_vectors_array


def update_proj_weights(eigen_values_array,
                        eigen_vectors_array,
                        T,
                        proj_vector:np.ndarray):
    
    """
    eigen_values_array-[N,2]
    eigen_vectors_array-[N,2,3]
    T-[4,4]坐标变换矩阵
    proj_vector-当前投影方向
    """

    T_roate = T.copy()
    T_roate[0,3] = 0
    T_roate[1,3] = 0
    T_roate[2,3] = 0


    N = len(eigen_values_array)
    #eigen_vectors_array_update-[4,N*2]
    eigen_vectors_array_update = np.column_stack( ( np.reshape(eigen_vectors_array,(2*N,3)),np.ones((2*N,1)) ) ).T
    eigen_vectors_array_update = T_roate@eigen_vectors_array_update
    #eigen_vectors_array_update-[N,2,3]([4,N*2]->[N*2,3]->[N,2,3])
    eigen_vectors_array_update = np.reshape(eigen_vectors_array_update[0:3].T,(N,2,3))
    weights = []

    for i in range(N):

        eigenvalues = eigen_values_array[i]
        lambda1 = eigenvalues[0]
        lambda2 = eigenvalues[1]

        eigenvectors = eigen_vectors_array_update[i]
        vector1 = eigenvectors[0]
        vector2 = eigenvectors[1]

        """计算当前投影方向与每个点的最佳投影方向之间的cos的绝对值,并将其作为权重进行输出"""
        weight = lambda1/(lambda1+lambda2) * np.linalg.norm(np.cross(vector1, proj_vector)) + lambda2/(lambda1+lambda2) * np.linalg.norm(np.cross(vector2, proj_vector))

        weights.append(weight)

    weights =  np.array(weights)
    return weights


def get_proj_weights(skele_points:np.ndarray, num_adj_points:int, proj_vector:np.ndarray):

    """
    Input:
    skele_points-血管的三维中心线点
    graph-中心线点组成的拓扑图
    num_adj_points-用于计算最佳投影角度的近邻点个数
    proj_vector-当前DSA机器的投影方向
    """
    N = skele_points.shape[1]
    weights = []
    
    for i in range(N):
        """对每个点取其近邻点num_adj_points个"""
        point = np.expand_dims(skele_points[0:3,i],axis=1)
        dists = np.sum((skele_points[0:3]-point)**2,axis=0)


        adj_point_indices = np.argsort(dists)[0:num_adj_points]

        #adj_points-[3,num_adj_points]
        adj_points = skele_points[0:3,adj_point_indices.tolist()]

        """计算近邻点集的最佳投影方向"""
        pca = PCA(n_components=2)
        pca.fit(adj_points.T)
        eigenvectors = pca.components_
        eigenvalues = pca.explained_variance_

        vector1 = eigenvectors[0]
        vector2 = eigenvectors[1]

        lambda1 = eigenvalues[0]
        lambda2 = eigenvalues[1]

        """计算当前投影方向与每个点的最佳投影方向之间的cos的绝对值,并将其作为权重进行输出"""
        weight = lambda1/(lambda1+lambda2) * np.linalg.norm(np.cross(vector1, proj_vector)) + lambda2/(lambda1+lambda2) * np.linalg.norm(np.cross(vector2, proj_vector))

        weights.append(weight)

    weights =  np.array(weights)
    return weights



if __name__=="__main__":


    """读取三维影像"""
    #读取影像数据
    dsa3d_seg_path = "F:\\3D_2D\\477583XA20201113\\3D_DSA_seg.nii.gz"
    seg_ia_img_dsa3d = sitk.ReadImage(dsa3d_seg_path)
    seg_ia_array_dsa3d = sitk.GetArrayFromImage(seg_ia_img_dsa3d).transpose(2,1,0)
    space_3d = seg_ia_img_dsa3d.GetSpacing()
    size_3d = seg_ia_img_dsa3d.GetSize()

    #生成T_3d_voxel_to_img
    T_3d_voxel_to_image_dsa3d = np.array([[space_3d[0],0          ,0          ,-(size_3d[0]-1)*space_3d[0]/2],
                                          [0          ,space_3d[1],0          ,-(size_3d[1]-1)*space_3d[1]/2],
                                          [0          ,0          ,space_3d[2],-(size_3d[2]-1)*space_3d[2]/2],
                                          [0          ,0          ,0          ,1                            ]])


    w0 = np.array([-np.pi/2,0,0,0,0,0])
    T0 = w_to_T(w0)
    """获取血管中心线的三维坐标"""
    #细化三维血管
    skeleton_3d = skeletonize_3d(seg_ia_array_dsa3d)
    #获取中心线三维坐标
    skeleton_3d_xyz = np.argwhere(skeleton_3d>0)#[N,3]
    s1_0 = np.column_stack((skeleton_3d_xyz,np.ones((len(skeleton_3d_xyz),1)))).T #[4,N]
    #将中心线三维坐标
    s1_0 = T0@T_3d_voxel_to_image_dsa3d@s1_0


    """获取投影方向"""
    dsa2d_ori_cor_path = "F:\\3D_2D\\477583XA20201113\\2D_DSA_cor.dcm"
    ori_img_dsa2d_cor, ori_array_dsa2d_cor, P_cor, image_plane_cor, dsa_param_cor = read_dsa_2d_ori(dsa2d_ori_cor_path)
    angle1 = dsa_param_cor[2]
    angle2 = dsa_param_cor[3]

    fig = plt.figure()
    ax = fig.add_subplot(111,projection="3d")

    ax_min = np.floor(s1_0[0:3].min(axis = 1)/10)*10
    ax_max = np.ceil(s1_0[0:3].max(axis = 1)/10)*10
        
    #显示的时候需要调转一下方向
    ax.set_xticks(np.linspace(ax_min[0], ax_max[0], int((ax_max[0]-ax_min[0])/5))+1)
    ax.set_yticks(np.linspace(ax_min[1], ax_max[1], int((ax_max[1]-ax_min[1])/5))+1)
    ax.set_zticks(np.linspace(ax_min[2], ax_max[2], int((ax_max[2]-ax_min[0])/5))+1)
    ax.set_box_aspect([ax_max[0] - ax_min[0],ax_max[1] - ax_min[1],ax_max[2] - ax_min[2]])


    # proj_vector = np.array([np.cos(angle2)*np.sin(angle1),
    #                         np.sin(angle2),
    #                         np.cos(angle2)*np.cos(angle1)])
    proj_vector = np.array([np.cos(angle2)*np.sin(angle1),
                            np.sin(angle2),
                            np.cos(angle2)*np.cos(angle1)])
    proj_weights = get_proj_weights(s1_0,50,proj_vector)
    ax.scatter(s1_0[0],s1_0[1],s1_0[2],c=proj_weights)
    ax.quiver(0,
              0,
              0,
              20*proj_vector[0],
              20*proj_vector[1],
              20*proj_vector[2])

    plt.show()
