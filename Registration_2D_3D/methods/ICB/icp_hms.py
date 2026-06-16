import SimpleITK as sitk
import numpy as np
from scipy.fftpack import fft,fftfreq
import torch
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize,skeletonize_3d
from scipy import optimize
from utils.transform_matrix import w_to_T
from sklearn.decomposition import PCA
import os
import time
import copy

from .proj_weight import get_proj_weights, get_eigen_list, update_proj_weights
from .topo_weight import get_topo_weights, create_graph, remove_noise_points, obtain_branches, get_point_indices_from_branches


def cartesian_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def polar_to_cartesian(r, theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)

    return np.array([x,y])

def correlate_theta(source_theta, target_theta):
    length = np.min([len(source_theta),len(target_theta)])
    corr = []
    for k in np.arange(-np.pi/4,np.pi/4,np.pi/180):
        source_theta_new = source_theta[0:length]+k
        
        corr.append(np.sum(np.abs(source_theta_new[0:length]-target_theta[0:length]))/length)

    return corr
def distance(w:np.ndarray,P:np.ndarray,x:np.ndarray,y:np.ndarray):

    num_points = np.min([x.shape[1],y.shape[1]])
    x_optim = x[:,0:num_points]
    y_optim = y[:,0:num_points]

    T = w_to_T(w)

    x_proj = func(P,T,x_optim)
    mean_distance = np.mean(np.sqrt((y_optim[0]-x_proj[0])**2+(y_optim[1]-x_proj[1])**2))

    return mean_distance

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



def adaptive_error(w:np.ndarray,
                   P:np.ndarray,
                   x:np.ndarray,
                   y:np.ndarray,
                   weights:np.ndarray):

    num_points = np.min([x.shape[1],y.shape[1]])
    x_optim = x[:,0:num_points]
    y_optim = y[:,0:num_points]
    weights_optim = weights[0:num_points]
    T = w_to_T(w)

    x_proj = func(P,T,x_optim)
    err = np.sum(weights_optim*((y_optim[0]-x_proj[0])**2+(y_optim[1]-x_proj[1])**2))

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


def branch_classification(skeleton_3d_xyz,branches_point_indices):

    branches_points = skeleton_3d_xyz[branches_point_indices]
    branches_class = []
    bifur = branches_points[0][-1]
    for branch in branches_points:
        start_point = branch[0]
        end_point = branch[-1]

"""计算两个二维点集之间的相似度"""
def DTW(p_source:np.ndarray, p_target:np.ndarray):

    """计算点集中各点之间的距离"""
    distance_matrix = []
    for i in range(p_source.shape[1]):
        distances = np.sum((np.expand_dims(p_source[:,i],axis=1)-p_target)**2, axis=0)
        distance_matrix.append(distances)
    distance_matrix = np.stack(distance_matrix,axis=0)

    """计算序列距离"""
    dtw_matrix = np.zeros([p_source.shape[1]+1, p_target.shape[1]+1])
    dtw_matrix[0,:] = np.inf
    dtw_matrix[:,0] = np.inf
    dtw_matrix[0,0] = 0

    for i in range(1,p_source.shape[1]+1):
        for j in range(1,p_target.shape[1]+1):
            dtw_matrix[i,j] = distance_matrix[i-1,j-1] + np.min([dtw_matrix[i-1,j],dtw_matrix[i,j-1],dtw_matrix[i-1,j-1]])


    """计算最短路径"""
    path_matrix = np.zeros_like(distance_matrix)
    i = distance_matrix.shape[0]-1
    j = distance_matrix.shape[1]-1
    path_matrix[i,j] = 1
    while (1):

        if i == 0 and j == 0:
            break

        flag = np.argmin([dtw_matrix[i+1-1,j+1],dtw_matrix[i+1,j+1-1],dtw_matrix[i+1-1,j+1-1]])

        if flag == 0:
            i = i-1
        elif flag == 1:
            j = j-1
        elif flag == 2:
            i = i-1
            j = j-1
        
        path_matrix[i,j] = 1

    return dtw_matrix[1:,1:], path_matrix

def similarity_DTW(p_source, p_target):

    """
    Input: 
    p_source-ndarray[2,N1]
    p_target-ndarray[2,N2]
    """

    """投影p_source"""
    #以起点为原点，PCA最佳投影方向为y轴，垂直方向为x轴构建坐标系
    p_source_zero_norm = p_source-np.expand_dims(p_source[:,0],axis=-1)
    pca = PCA(n_components=1)
    pca.fit(p_source_zero_norm.T)
    x_axis_source = pca.components_[0]
    fractor_source = np.sqrt(np.sum(x_axis_source**2))
    y_axis_source = np.array([-x_axis_source[1]/fractor_source, 
                               x_axis_source[0]/fractor_source])
    
    #投影
    T_source = np.stack([x_axis_source,y_axis_source])
    p_source_norm = T_source@p_source_zero_norm


    """投影p_target"""
    #以起点为原点，PCA最佳投影方向为y轴，垂直方向为x轴构建坐标系
    p_target_zero_norm = p_target-np.expand_dims(p_target[:,0],axis=-1)
    pca = PCA(n_components=1)
    pca.fit(p_target_zero_norm.T)
    x_axis_target = pca.components_[0]
    fractor_target = np.sqrt(np.sum(x_axis_target**2))
    y_axis_target = np.array([-x_axis_target[1]/fractor_target, 
                               x_axis_target[0]/fractor_target])

    #投影
    T_target = np.stack([x_axis_target,y_axis_target])
    p_target_norm = T_target@p_target_zero_norm



    #对曲线进行旋转对齐
    r_source, theta_source = cartesian_to_polar(p_source_norm[0],p_source_norm[1])
    r_target, theta_target = cartesian_to_polar(p_target_norm[0],p_target_norm[1])
    theta_corr = correlate_theta(theta_source,theta_target)
    theta_cali = np.argmin(theta_corr)*np.pi/180 - np.pi/4
    if np.min(theta_corr) > 0.1:
        theta_cali = 0
    p_source_norm_rot = polar_to_cartesian(r_source, theta_source+theta_cali)
    

    """对两组序列进行全局匹配"""
    #计算source的傅里叶变换
    # x_source_w = fft(p_source_norm[0])
    # x_source_w_abs = np.abs(x_source_w)/len(x_source_w)
    # x_source_w_phase = np.angle(x_source_w)

    y_source_w = fft(p_source_norm[1])
    y_source_w_abs = np.abs(y_source_w)/len(y_source_w)
    y_source_w_phase = np.angle(y_source_w)

    #计算target的傅里叶变换
    # x_target_w = fft(p_target_norm[0])
    # x_target_w_abs = np.abs(x_target_w)/len(x_target_w)
    # x_target_w_phase = np.angle(x_target_w)
    
    y_target_w = fft(p_target_norm[1])
    y_target_w_abs = np.abs(y_target_w)/len(y_target_w)
    y_target_w_phase = np.angle(y_target_w)


    w = np.arange(1,8)*2*np.pi
    phi_w = y_source_w_phase[1:8]
    lambda_w = y_target_w_phase[1:8]

    
    t_cali = int(np.mean((phi_w-lambda_w)/w)*len(y_source_w))


    if t_cali >= 0:
        p_source_norm_cali = p_source_norm_rot
        p_target_norm_cali = p_target_norm[:,t_cali:]-np.expand_dims(p_target_norm[:,t_cali],axis=-1)
    else:
        p_source_norm_cali = p_source_norm[:,-t_cali:]-np.expand_dims(p_source_norm[:,-t_cali],axis=-1)
        p_target_norm_cali = p_target_norm
    

    # #计算互相关
    # corr = np.correlate(p_source_norm_rot[1], p_target_norm[1], mode='full')
    # #时延补偿
    # t_cali = (np.argmax(corr)-len(corr)//2)//2
    # if t_cali <= 0:
    #     p_source_norm_cali = p_source_norm_rot
    #     p_target_norm_cali = p_target_norm[:,-t_cali:]-np.expand_dims(p_target_norm[:,-t_cali],axis=-1)
    # elif t_cali > 0:
    #     p_source_norm_cali = p_source_norm_rot[:,t_cali:]-np.expand_dims(p_source_norm_rot[:,t_cali],axis=-1)
    #     p_target_norm_cali = p_target_norm


    # #计算互相关
    # _,N = p_source_norm.shape
    # _,M = p_target_norm.shape
    # corr = np.correlate(p_source_norm[0], p_target_norm[0], mode='full') + np.correlate(p_source_norm[1], p_target_norm[1], mode='full')

    # #时延补偿
    # t_cali = (np.argmax(corr)-(M-1))//2

    # if t_cali <= 0 and t_cali >= -(M-1)//3:
    #     p_source_norm_cali = p_source_norm
    #     p_target_norm_cali = p_target_norm[:,-t_cali:]-np.expand_dims(p_target_norm[:,-t_cali],axis=-1)

    # elif t_cali > 0 and t_cali <= (N-1)//3:
    #     p_source_norm_cali = p_source_norm[:,t_cali:]-np.expand_dims(p_source_norm[:,t_cali],axis=-1)
    #     p_target_norm_cali = p_target_norm

    # else:
    #     p_source_norm_cali = p_source_norm
    #     p_target_norm_cali = p_target_norm




    """对两组序列进行局部校准"""
    dtw_matrix, similarity_matrix_of_normed_squeence = DTW(p_source_norm_cali,p_target_norm_cali)
    similarity_matrix = np.zeros([p_source.shape[1],p_target.shape[1]])
    row_cali = similarity_matrix.shape[0]-similarity_matrix_of_normed_squeence.shape[0]
    colum_cali = similarity_matrix.shape[1]-similarity_matrix_of_normed_squeence.shape[1]
    similarity_matrix[row_cali:,colum_cali:] = similarity_matrix_of_normed_squeence


    """将局部校准后尾部多余的数据去除"""
    for k in range(1,5):
        if np.sum(similarity_matrix[:,-k])>10:
            end_index_row = np.min(np.argwhere(similarity_matrix[:,-k]==1))
            for j in range(1,k+1):
                similarity_matrix[:,-j] = 0
            similarity_matrix[end_index_row, -k] = 1

        elif np.sum(similarity_matrix[-k,:])>10:
            end_index_colum = np.min(np.argwhere(similarity_matrix[-k,:]==1))
            for j in range(1,k+1):
                similarity_matrix[-j,:] = 0
            similarity_matrix[-k, end_index_colum] = 1



    """显示"""
    # fig2 = plt.figure()

    # fig2_ax1 = fig2.add_subplot(221)
    # fig2_ax1.set_xlim(np.min(p_source_norm[1]),np.max(p_source_norm[1]))
    # fig2_ax1.set_ylim(np.min(p_source_norm[0]),np.max(p_source_norm[0]))
    # fig2_ax1.plot(p_source_norm[1],p_source_norm[0])


    # fig2_ax2 = fig2.add_subplot(222)
    # fig2_ax2.imshow(dtw_matrix)

    # fig2_ax3 = fig2.add_subplot(223)
    # fig2_ax3.imshow(similarity_matrix)

    # fig2_ax4 = fig2.add_subplot(224)
    # fig2_ax4.set_xlim(np.min(p_target_norm[0]),np.max(p_target_norm[0]))
    # fig2_ax4.set_ylim(np.min(p_target_norm[1]),np.max(p_target_norm[1]))
    # fig2_ax4.plot(p_target_norm[0],p_target_norm[1])

    # fig3 = plt.figure()

    # fig3_ax9 = fig3.add_subplot(221)
    # fig3_ax9.plot(-p_source_norm[1],-p_source_norm[0])
    # fig3_ax9.plot(-p_target_norm[1],-p_target_norm[0])

    # fig3_ax9 = fig3.add_subplot(223)
    # fig3_ax9.plot(-p_source_norm_cali[1],-p_source_norm_cali[0])
    # fig3_ax9.plot(-p_target_norm_cali[1],-p_target_norm_cali[0])


    # fig3_ax10 = fig3.add_subplot(224)
    # # fig3_ax10.plot(np.arange(-(M-1),N),corr)
    # fig3_ax10.plot(np.arange(len(corr)),corr)

    # fig4 = plt.figure()

    # fig4_ax1 = fig4.add_subplot(221)
    # fig4_ax1.plot(-p_source_norm[1],-p_source_norm[0])
    # fig4_ax1.plot(-p_target_norm[1],-p_target_norm[0])

    # fig4_ax2 = fig4.add_subplot(222)
    # fig4_ax2.plot(-p_source_norm_rot[1],-p_source_norm_rot[0])
    # fig4_ax2.plot(-p_target_norm[1],-p_target_norm[0])

    # fig4_ax3 = fig4.add_subplot(223)
    # fig4_ax3.plot(-p_source_norm_cali[1],-p_source_norm_cali[0])
    # fig4_ax3.plot(-p_target_norm_cali[1],-p_target_norm_cali[0])

    # fig4_ax4 = fig4.add_subplot(224)
    # fig4_ax4.plot(np.arange(-np.pi/4,np.pi/4,np.pi/180),theta_corr)

    # fig5 = plt.figure()
    # fig5_ax1 = fig5.add_subplot(2,2,1)
    # fig5_ax1.plot(np.arange(len(y_source_w_abs)//2)*2*np.pi,y_source_w_abs[0:len(y_source_w_abs)//2])
    # fig5_ax2 = fig5.add_subplot(2,2,2)
    # fig5_ax2.plot(np.arange(len(y_source_w_phase)//2)*2*np.pi,y_source_w_phase[0:len(y_source_w_phase)//2])

    # fig5_ax3 = fig5.add_subplot(2,2,3)
    # fig5_ax3.plot(np.arange(len(y_target_w_abs)//2)*2*np.pi,y_target_w_abs[0:len(y_target_w_abs)//2])
    # fig5_ax4 = fig5.add_subplot(2,2,4)
    # fig5_ax4.plot(np.arange(len(y_target_w_phase)//2)*2*np.pi,y_target_w_phase[0:len(y_target_w_phase)//2])



    # plt.show()




    return similarity_matrix

def branch_similarity_DTW(source:list[np.ndarray], target:list[np.ndarray]):
    """
    Input:
    source - list[ndarray[C+1,K1],ndarray[C+1,K2],...]
    target - list[ndarray[C+1,K1'],ndarray[C+1,K2'],...]
    Output:
    branches_similarity_matrixes - list[ndarray[K1,K1'], ndarray[K1,K1']]
    """
    num_branches = min([len(source), len(target)])
    branches_similarity_matrixes = []

    for i in range(num_branches):
        similarity_matrix = similarity_DTW(source[i][0:-1], target[i][0:-1])
        branches_similarity_matrixes.append(similarity_matrix)



    return branches_similarity_matrixes

"""常规相似性测度"""
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

    similarity_matrix = np.zeros(shape=(p_source.shape[0],
                                        p_target.shape[0]))

    for i,p in enumerate(p_source):
        distance = np.sum((np.expand_dims(p,axis=0)-p_target)**2,axis=-1)
        min_id = np.argmin(distance)
        similarity_matrix[i,min_id] = 1

        # count[min_id] = count[min_id]+1
        # if count[min_id] > 1:
        #     distance_mask[min_id] = 10000000

    return similarity_matrix  

def branch_similarity(source:list[np.ndarray], target:list[np.ndarray]):
    """
    Input:
    source - list[ndarray[C,K1],ndarray[C,K2],...]
    target - list[ndarray[C,K1'],ndarray[C,K2'],...]
    Output:
    branches_similarity_matrixes - list[ndarray[K1,K1'], ndarray[K1,K1']]
    """
    num_branches = min([len(source), len(target)])
    branches_similarity_matrixes = []



    for i in range(num_branches):
        similarity_matrix = similarity(source[i], target[i])
        branches_similarity_matrixes.append(similarity_matrix)

    return branches_similarity_matrixes

# def branch_similarity_modify(source:list[np.ndarray], target:list[np.ndarray]):
#     """
#     假设source是三维的血管中心线，一共有三个分支，而二维血管中心线可能有三个分支，可能只有一个分支
#     Input:
#     source - list[ndarray[C,K1],ndarray[C,K2],...]
#     target - list[ndarray[C,K1'],ndarray[C,K2'],...]
#     Output:
#     branches_similarity_matrixes - list[ndarray[K1,K1'], ndarray[K1,K1']]
#     """

#     branches_similarity_matrixes = []
#     if len(target) == 1:
#         """计算单独的相似度"""
#         source_0 = source[0]

#         """计算和branch1合并后的相似度"""
#         source_1 = np.concatenate([source[0],source[1]])

#         """计算和branch2合并后的相似度"""
#         source_2 = np.concatenate([source[0],source[2]])

#     elif len(target) == 3:
#         source_0 = [source[0],source[1],source[2]]
#         source_1 = [source[0],source[2],source[1]]

#     else:
#         num_branches = min([len(source), len(target)])
#         for i in range(num_branches):
#             similarity_matrix = similarity(source[i], target[i])
#             branches_similarity_matrixes.append(similarity_matrix)

#     return branches_similarity_matrixes

def get_branches_near_points(source:list[np.ndarray],
                             target:list[np.ndarray], 
                             branches_similarity_matrixes:list[np.ndarray]):
    
    """
    Input:
    source - list[ndarray[C,K1],ndarray[C,K2],...]
    target - list[ndarray[C,K1'],ndarray[C,K2'],...]
    branches_similarity_matrixes - list[ndarray[K1,K1'],ndarray[K2,K2'],...]
    Output:
    branches_near_points - list[ndarray[C,K1],ndarray[C,K2],...]
    """
    branches_near_points = []

    for i in range(len(branches_similarity_matrixes)):
        source_branch = source[i]
        target_branch = target[i]
        branch_similarity_matrix = branches_similarity_matrixes[i]
        branch_near_points = []
        for j in range(branch_similarity_matrix.shape[0]):
            near_point_index = np.argwhere(branch_similarity_matrix[j]==1)

            #若有近邻点，则将其加入到近邻点集中
            if near_point_index.size > 0:
                branch_near_points.append(target_branch[:,near_point_index[0][0]])
            #若无近邻点，则将其投影加入到近邻点集中
            else:
                branch_near_points.append(source_branch[:,j])
        branch_near_points = np.array(branch_near_points).T
        branches_near_points.append(branch_near_points)

    return branches_near_points


def get_branch_points(points, branches_points_indices):
    """
    Input:
    points-ndarray[C,N]
    branches_points_indices-list[list[K1],list[K2],...]
    Return:
    branches_points-list[ndarray[C,K1],ndarray[C,K2],...]
    """

    branches_points = []
    for branch in branches_points_indices:
        branches_points.append(points[:,branch])
    
    return branches_points


def branch_classify(center,branch_points,branch_points_indices,branch_nodes_indices):
    """
    branch_points-[ndarray[C+1,N1],ndarray[C+,N2],...]
    """
    if len(branch_points) == 3:
        #获取分叉点
        bifur = branch_points[0][0:-1,-1]
        bifur_direction = bifur-center
        #计算第一个分支的方向
        branch_direction1 = np.mean(branch_points[1][0:-1],axis=1)-bifur
        branch_direction2 = np.mean(branch_points[2][0:-1],axis=1)-bifur
        if bifur_direction@branch_direction1<0 and bifur_direction@branch_direction2>0:
            branch_points[1:3] = branch_points[::-1][0:2]
            branch_points_indices[1:3] = branch_points_indices[::-1][0:2]
            branch_nodes_indices[1:3] = branch_nodes_indices[::-1][0:2]

    return branch_points,branch_points_indices,branch_nodes_indices

def ICP_HMS(array_3d,
                                           array_2d_cor, 
                                           array_2d_sag,
                                           T_3d_voxel_to_img,
                                           w_0,
                                           P_cor,
                                           angles_cor,
                                           P_sag,
                                           angles_sag,
                                           err_limit=0.01):
    """
    Input: 
    array_3d-[H,W,D]
    array_2d_cor-[H,W]
    array_2d_sag-[H,W]w_0
    Output: 
    T-[4,4],三维数据的坐标变换矩阵
    """
    time_flag_0 = time.time()

    """获取血管中心线的三维坐标"""
    #细化三维血管
    skele_3d_volume = skeletonize_3d(array_3d)
    #获取中心线三维坐标(体素坐标)
    skele_3d_points_voxel = np.argwhere(skele_3d_volume>0)#[N,3]
    skele_3d_points_voxel = np.column_stack((skele_3d_points_voxel,np.ones((len(skele_3d_points_voxel),1)))).T #[4,N]
    #获取中心线三维坐标(图像坐标系)
    skele_3d_points_volume = T_3d_voxel_to_img@skele_3d_points_voxel
    #构造graph
    skele_3d_nodes = create_graph(skele_3d_points_volume, np.argmin(skele_3d_points_volume[2]))
    remove_noise_points(skele_3d_nodes,20)
    #划分branch
    branches_3d_nodes_indices = []
    obtain_branches(skele_3d_nodes,0,[],branches_3d_nodes_indices)
    branches_3d_points_indices_1 = get_point_indices_from_branches(skele_3d_nodes,branches_3d_nodes_indices)
    branches_3d_points_volume = get_branch_points(skele_3d_points_volume, branches_3d_points_indices_1)
    #调整分支顺序：C1，A1，M1
    branches_3d_points_volume,branches_3d_points_indices_1,branches_3d_nodes_indices = branch_classify(np.zeros(3),
                                                                                                       branches_3d_points_volume,
                                                                                                       branches_3d_points_indices_1,
                                                                                                       branches_3d_nodes_indices)


    time_flag_1 = time.time()



    """获取血管中心线在coronal平面的像素坐标"""
    #细化二维血管
    skele_2d_cor_image = skeletonize(array_2d_cor)
    #获取中心线二维坐标
    skele_2d_cor_points_pixel = np.argwhere(skele_2d_cor_image>0)[:,0:2]
    skele_2d_cor_points_pixel = np.column_stack((skele_2d_cor_points_pixel,np.ones((len(skele_2d_cor_points_pixel),1)))).T #[3,N]
    #构造Graph,剪掉外点
    skele_2d_cor_nodes = create_graph(skele_2d_cor_points_pixel,np.argmax(skele_2d_cor_points_pixel[1]))
    remove_noise_points(skele_2d_cor_nodes,50)
    #划分branch
    branches_2d_cor_nodes_indices = []
    obtain_branches(skele_2d_cor_nodes,0,[],branches_2d_cor_nodes_indices)
    branches_2d_cor_points_indices = get_point_indices_from_branches(skele_2d_cor_nodes, branches_2d_cor_nodes_indices)
    branches_2d_cor_points_pixel = get_branch_points(skele_2d_cor_points_pixel, branches_2d_cor_points_indices)
    #调整分支顺序：C1，A1，M1
    branches_2d_cor_points_pixel,branches_2d_cor_points_indices,branches_2d_cor_nodes_indices = branch_classify(np.array(array_2d_cor.shape[0:2])/2,
                                                                                                                branches_2d_cor_points_pixel,
                                                                                                                branches_2d_cor_points_indices,
                                                                                                                branches_2d_cor_nodes_indices)



    """获取血管中心线在saggital平面的像素坐标"""
    #细化二维血管
    skele_2d_sag_image = skeletonize(array_2d_sag)
    #获取中心线二维坐标
    skele_2d_sag_points_pixel = np.argwhere(skele_2d_sag_image>0)[:,0:2]
    skele_2d_sag_points_pixel = np.column_stack((skele_2d_sag_points_pixel,np.ones((len(skele_2d_sag_points_pixel),1)))).T #[3,N]
    #构造Graph,剪掉外点
    skele_2d_sag_nodes = create_graph(skele_2d_sag_points_pixel,np.argmax(skele_2d_sag_points_pixel[1]))
    remove_noise_points(skele_2d_sag_nodes,50)
    #划分branch
    branches_2d_sag_nodes_indices = []
    obtain_branches(skele_2d_sag_nodes,0,[],branches_2d_sag_nodes_indices)
    branches_2d_sag_nodes_indices = [branches_2d_sag_nodes_indices[0]]#侧位只要C1分支
    branches_2d_sag_points_indices = get_point_indices_from_branches(skele_2d_sag_nodes,branches_2d_sag_nodes_indices)
    branches_2d_sag_points_pixel = get_branch_points(skele_2d_sag_points_pixel, branches_2d_sag_points_indices)

    time_flag_2 = time.time()

    # """显示1/3"""
    # fig = plt.figure()
    # ax_3d_cor = fig.add_subplot(221,projection="3d")
    # ax_cor = fig.add_subplot(222)
    # ax_3d_sag = fig.add_subplot(223,projection="3d")
    # ax_sag = fig.add_subplot(224)

    # ax_min = np.floor(skele_3d_points_volume[0:3].min(axis = 1)/10)*10
    # ax_max = np.ceil(skele_3d_points_volume[0:3].max(axis = 1)/10)*10
        
    # #正位
    # ax_3d_cor.set_xticks(np.linspace(ax_min[0], ax_max[0], int((ax_max[0]-ax_min[0])/5))+1)
    # ax_3d_cor.set_yticks(np.linspace(ax_min[1], ax_max[1], int((ax_max[1]-ax_min[1])/5))+1)
    # ax_3d_cor.set_zticks(np.linspace(ax_min[2], ax_max[2], int((ax_max[2]-ax_min[2])/5))+1)
    # ax_3d_cor.set_box_aspect([ax_max[0] - ax_min[0],ax_max[1] - ax_min[1],ax_max[2] - ax_min[2]])

    # ax_cor.set_xlim(0,array_2d_cor.shape[0])
    # ax_cor.set_ylim(array_2d_cor.shape[1],0)
    # ax_cor.set_box_aspect(1)

    # ax_sag.set_xlim(0,array_2d_sag.shape[0])
    # ax_sag.set_ylim(array_2d_sag.shape[1],0)
    # ax_sag.set_box_aspect(1)

    # for branch_3d in branches_3d_points_volume:
    #     ax_3d_cor.scatter(branch_3d[0],
    #                       branch_3d[1],
    #                       branch_3d[2])


    # plt.show()


    """计算血管中心线各点处的最佳投影方向和DSA机器的正侧位投影方向"""
    #获取各点处的最佳投影平面
    eigen_values_list, eigen_vectors_list = get_eigen_list(skele_3d_points_volume,50)

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


    """迭代优化"""
    distance_cor_old = None
    distance_sag_old = None
    w_old = w_0
    w = w_0
    T = w_to_T(w_0)
    i = 0

    modify_flag = 0
    len0 = len(branches_3d_points_indices_1[0])

    time_flag_3 = time.time()

    while(1):

        """根据T对血管中心线的原始三维坐标进行调整"""
        time_flag_4 = time.time()
        s1 = T@skele_3d_points_volume


        # """显示2/3"""
        # ax_cor.clear()
        # ax_sag.clear()

        # ax_cor.set_xlim(0,array_2d_cor.shape[0])
        # ax_cor.set_ylim(array_2d_cor.shape[1],0)
        # ax_cor.set_box_aspect(1)

        # ax_sag.set_xlim(0,array_2d_sag.shape[0])
        # ax_sag.set_ylim(array_2d_sag.shape[1],0)
        # ax_sag.set_box_aspect(1)

        # for branch_2d_cor in branches_2d_cor_points_pixel:
        #     ax_cor.scatter(branch_2d_cor[0],
        #                    branch_2d_cor[1],)
        # for branch_2d_sag in branches_2d_sag_points_pixel:
        #     ax_sag.scatter(branch_2d_sag[0],
        #                    branch_2d_sag[1],)

        """更新权重"""
        #更新正位权重
        # proj_weights_cor = update_proj_weights(eigen_values_list,
        #                                        eigen_vectors_list,
        #                                        T,
        #                                        proj_vector_cor)
        weights_cor = np.ones(len(eigen_values_list))

        #更新侧位权重
        # proj_weights_sag = update_proj_weights(eigen_values_list,
        #                                        eigen_vectors_list,
        #                                        T,
        #                                        proj_vector_sag)
        weights_sag = np.ones(len(eigen_values_list))



        """计算相似度,并进行点匹配"""
        s1_p_cor = P_cor@s1
        s1_p_cor = s1_p_cor/np.expand_dims(s1_p_cor[2],axis=0)
        branches_proj_cor_points_pixel = get_branch_points(s1_p_cor,branches_3d_points_indices_1)
        branches_cor_similarity_matrixes = branch_similarity_DTW(branches_proj_cor_points_pixel,branches_2d_cor_points_pixel)
        branches_2d_cor_near_points_pixel = get_branches_near_points(branches_proj_cor_points_pixel,
                                                                     branches_2d_cor_points_pixel,
                                                                     branches_cor_similarity_matrixes)



        
        s1_p_sag = P_sag@s1
        s1_p_sag = s1_p_sag/np.expand_dims(s1_p_sag[2],axis=0)
        branches_proj_sag_points_pixel = get_branch_points(s1_p_sag,branches_3d_points_indices_1)
        branches_sag_similarity_matrixes = branch_similarity_DTW(branches_proj_sag_points_pixel,branches_2d_sag_points_pixel)
        branches_2d_sag_near_points_pixel = get_branches_near_points(branches_proj_sag_points_pixel,
                                                                     branches_2d_sag_points_pixel,
                                                                     branches_sag_similarity_matrixes)

        # """显示3/3"""
        # for branch_proj_2d_cor in branches_proj_cor_points_pixel:
        #     ax_cor.scatter(branch_proj_2d_cor[0],
        #                    branch_proj_2d_cor[1],)
        # for branch_proj_2d_sag in branches_proj_sag_points_pixel:
        #     ax_sag.scatter(branch_proj_2d_sag[0],
        #                    branch_proj_2d_sag[1],)

        # for branch_show_index_cor in range(len(branches_proj_cor_points_pixel)):

        #     if branch_show_index_cor >= len(branches_2d_cor_near_points_pixel):
        #         break
        #     start_points_cor = branches_proj_cor_points_pixel[branch_show_index_cor][:,::10]
        #     end_points_cor = branches_2d_cor_near_points_pixel[branch_show_index_cor][:,::10]

        #     for start,end in zip(start_points_cor.T,end_points_cor.T):
        #         ax_cor.arrow(start[0],
        #                      start[1],
        #                      end[0]-start[0],
        #                      end[1]-start[1])

        # for branch_show_index_sag in range(len(branches_proj_sag_points_pixel)):

        #     if branch_show_index_sag >= len(branches_2d_sag_near_points_pixel):
        #         break
        #     start_points_sag = branches_proj_sag_points_pixel[branch_show_index_sag][:,::10]
        #     end_points_sag = branches_2d_sag_near_points_pixel[branch_show_index_sag][:,::10]

        #     for start,end in zip(start_points_sag.T,end_points_sag.T):
        #         ax_sag.arrow(start[0],
        #                      start[1],
        #                      end[0]-start[0],
        #                      end[1]-start[1])


        s1_0_branches_flatten = np.concatenate(branches_3d_points_volume,axis=1)
        near_point_cor_flatten = np.concatenate(branches_2d_cor_near_points_pixel,axis=1)
        near_point_sag_flatten = np.concatenate(branches_2d_sag_near_points_pixel,axis=1)


        """进行优化更新T"""
        problem = optimize.OptimizeResult(
            fun = adaptive_error_two_view,
            x0 = w,
            args = (s1_0_branches_flatten,
                    P_cor,near_point_cor_flatten,weights_cor,
                    P_sag,near_point_sag_flatten,weights_sag),
            # bounds = [(-10000,10000),(-10000,10000),(-10000,10000),
            #           (-500,500),(-500,500),(-500,500)]
        )
        optim_result = optimize.minimize(**problem)
        w = optim_result.x
        


        # obj_err = optim_result.fun
        distance_cor = distance(w,P_cor,s1_0_branches_flatten,near_point_cor_flatten)
        distance_sag = distance(w,P_sag,s1_0_branches_flatten,near_point_sag_flatten)
        print("Epoch {} error cor:{}".format(i,distance_cor))
        print("Epoch {} error sag:{}".format(i,distance_sag))

        """计算前一次位姿和当前位姿的偏移量,过小则终止循环"""
        if distance_cor_old is not None and distance_sag_old is not None:
            err = np.sum(np.abs(distance_cor_old-distance_cor)+np.abs(distance_sag_old-distance_sag))        
            if err < err_limit or i > 20:

                if modify_flag == 2:
                    break

                if distance_cor > 30 or distance_sag > 30:
                    if len(branches_2d_cor_points_pixel) > 1 and len(branches_3d_points_indices_1) == 3:
                        branches_3d_points_volume[1:3] = branches_3d_points_volume[::-1][0:2]
                        branches_3d_points_indices_1[1:3] = branches_3d_points_indices_1[::-1][0:2]
                        w = w_0
                        i = 0

                        modify_flag = 2

                    elif len(branches_2d_cor_points_pixel) == 1 and len(branches_3d_points_indices_1) > 3:
                        #这里要按顺序来试，第一个branch和第二个branch合并, 第一个branch和第三个branch合并

                        if modify_flag == 0:
                            branches_3d_points_indices_1 = [np.concatenate([branches_3d_points_indices_1[0],
                                                                            branches_3d_points_indices_1[1]]), 
                                                            branches_3d_points_indices_1[2]]
                            branches_3d_points_volume = [np.concatenate([branches_3d_points_volume[0],
                                                                         branches_3d_points_volume[1]],axis=1), 
                                                            branches_3d_points_volume[2]]
                            modify_flag = 1
                            w = w_0
                            i = 0
                        elif modify_flag == 1:
                            branches_3d_points_indices_1 = [np.concatenate([branches_3d_points_indices_1[0][0:len0],
                                                                            branches_3d_points_indices_1[1]]), 
                                                            branches_3d_points_indices_1[0][len0:]]
                            branches_3d_points_volume = [np.concatenate([branches_3d_points_volume[0][0:len0],
                                                                         branches_3d_points_volume[1]],axis=1), 
                                                            branches_3d_points_volume[0][len0:]]
                            w = w_0
                            i = 0

                            modify_flag = 2
                    else:
                        break
                else:
                    break
        # if w_old is not None:
        #     err = np.sum((np.array(w)-np.array(w_old))**2)       
        #     if err < err_limit:
        #         if distance_cor > 30 or distance_sag > 30:
        #             if len(branches_3d_points_indices_1) == 3:
        #                 branches_3d_points_volume[1:3] = branches_3d_points_volume[::-1][0:2]
        #                 branches_3d_points_indices_1[1:3] = branches_3d_points_indices_1[::-1][0:2]
        #             else:
        #                 break


        # """更新"""
        T = w_to_T(w)
        distance_cor_old = distance_cor
        distance_sag_old = distance_sag
        w_old = w
        i = i+1

    time_flag_5 = time.time()
    print(time_flag_5-time_flag_1)
    
    print("registration done!")
    return w,T,distance_cor,distance_sag