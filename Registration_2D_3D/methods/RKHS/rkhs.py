import numpy as np
from scipy import optimize
import json

def read_dsa_2d_seg(seg_path):
    
    #读取冠状面二维分割图
    dsa_seg_2d_img = sitk.ReadImage(seg_path)
    dsa_seg_2d_array = sitk.GetArrayFromImage(dsa_seg_2d_img).transpose(2,1,0)
    dsa_seg_2d_array = np.ascontiguousarray(dsa_seg_2d_array)

    return dsa_seg_2d_img, dsa_seg_2d_array

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

def gaussian_kernal(control_point,moving_point,sigma):
    
    """
    Input:
    control_point-[3,N2],控制点
    moving_point-[3,N1],待配准的点
    sigma-[1,N2],控制点上高斯核的尺寸
    Return:
    kernel_matrix-[N2,N1]
    """

    """计算每个moving_points到每个control_points的距离"""
    kernal_matrix = []
    for i in range(control_point.shape[1]):
        dist = np.sum((moving_point-np.expand_dims(control_point[:,i],axis=1))**2,axis=0)
        kernal_matrix.append(dist)

    kernal_matrix = np.stack(kernal_matrix,axis=0)

    """计算高斯函数"""
    kernal_matrix = np.exp(-kernal_matrix/(2*sigma**2))
    
    return kernal_matrix



def RKHS_norm(weight:np.ndarray,
              control_point:np.ndarray,
              sigma:np.ndarray):
    """
    Input:
    weight-[3,N2]
    control_point-[3,N2]
    Output:

    """

    #U-[N2,N2]
    U = gaussian_kernal(control_point,control_point,sigma)
    rkhs_norm = weight@U@weight.T

    return rkhs_norm.trace()

def Tikhonov_regularization_distance(weight, 
                                     P, 
                                     control_point_3d, 
                                     moving_point_3d, 
                                     target_point_2d, 
                                     sigma):
    """
    Input:
    weight-[3,N2]
    P-[3,4]
    control_point_3d-[3,N2]
    moving_point_3d-[3,N1]
    target_point_2d-[2,N1]
    sigma-[N2],控制点上高斯核的尺寸
    """


    """获取高斯核函数的值-[N2,N1]"""
    kernal_matrix = gaussian_kernal(control_point_3d,moving_point_3d,sigma)

    """计算距离损失"""
    #计算位移之后的向量-[3,N1]
    moving_point_3d_transformation = weight@kernal_matrix

    #计算投影点
    moving_point_3d_transformation_homo = np.concatenate([moving_point_3d_transformation,
                                                          np.ones((1,moving_point_3d_transformation.shape[1]))],
                                                        axis=0)
    x_proj_homo = P@moving_point_3d_transformation_homo
    x_proj_homo = x_proj_homo/np.expand_dims(x_proj_homo[2],axis=0)
    x_proj = x_proj_homo[0:2]#

    #计算投影点和对应点之间的距离损失
    distance_loss = np.sum((x_proj-target_point_2d)**2)/x_proj.shape[1]


    return distance_loss

def Tikhonov_regularization_distance_gradient(weight, 
                                              P, 
                                              control_point_3d, 
                                              moving_point_3d, 
                                              target_point_2d, 
                                              sigma):

    """求第一层导数: 平方距离均值对moving points配准后的坐标矩阵的导数"""
    #求配准后的坐标矩阵T-[3,N1]
    kernal_matrix = gaussian_kernal(control_point_3d,moving_point_3d,sigma)
    T = weight@kernal_matrix
    T_homo = np.concatenate([T,np.ones((1,T.shape[1]))],axis=0)
    #求投影后的坐标矩阵P_T-[2,N1]
    P_T_homo = P@T_homo
    P_T = P_T_homo[0:2]/np.expand_dims(P_T_homo[2],axis=0)

    #求配对点之间的偏移矩阵D-[2,N1]
    D = target_point_2d-P_T

    #求偏移向量对配准后坐标向量的雅可比矩阵J
    J = []
    K = moving_point_3d.shape[1]
    for k in range(K):
        J_k = [[P[j,i]*P[2,:]@T_homo[:,k] - P[2,i]*P[j,:]@T_homo[:,k] for j in range(2)] for i in range(3)]/(P[3,:]@T_homo[:,k])**2

        J.append(J_k)


    """求第二层导数: moving points向量矩阵对权重矩阵的导数"""

    pass

def Tikhonov_regularization_two_view(weight, 
                                     P_cor,
                                     P_sag, 
                                     control_point_3d, 
                                     moving_point_3d, 
                                     target_point_2d_cor,
                                     target_point_2d_sag, 
                                     sigma, 
                                     lambda_norm):
    """
    Input:
    weight-[3,N2]
    P-[3,4]
    control_point_3d-[3,N2]
    moving_point_3d-[3,N1]
    target_point_2d-[2,N1]
    sigma-[N2],控制点上高斯核的尺寸
    """

    weight = np.reshape(weight,(3,-1))

    regularization_cor = Tikhonov_regularization_distance(weight,
                                                 P_cor,
                                                 control_point_3d,
                                                 moving_point_3d,
                                                 target_point_2d_cor,
                                                 sigma)
    regularization_sag = Tikhonov_regularization_distance(weight,
                                                 P_sag,
                                                 control_point_3d,
                                                 moving_point_3d,
                                                 target_point_2d_sag,
                                                 sigma)
    
    rkhs_norm = RKHS_norm(weight,control_point_3d,sigma)

    return regularization_cor + regularization_sag + lambda_norm*rkhs_norm




if __name__ == "__main__":

    import os
    import SimpleITK as sitk
    from skimage.morphology import skeletonize,skeletonize_3d

    """获取三维点集"""

    data_dir = "F:/backup/data/IAS/IAS_2D_3D_Registration/using/IM_0005_label"
    #读取影像数据
    dsa3d_seg_path = data_dir+"/seg_3d.nii.gz"
    seg_ia_img_dsa3d = sitk.ReadImage(dsa3d_seg_path)
    seg_ia_array_dsa3d = sitk.GetArrayFromImage(seg_ia_img_dsa3d).transpose(2,1,0)
    space_3d = seg_ia_img_dsa3d.GetSpacing()
    size_3d = seg_ia_img_dsa3d.GetSize()

    #生成T_3d_voxel_to_img
    T_3d_voxel_to_image_dsa3d = np.array([[space_3d[0],0          ,0          ,-(size_3d[0]-1)*space_3d[0]/2],
                                              [0          ,space_3d[1],0          ,-(size_3d[1]-1)*space_3d[1]/2],
                                              [0          ,0          ,space_3d[2],-(size_3d[2]-1)*space_3d[2]/2],
                                              [0          ,0          ,0          ,1                            ]])

    w0 = np.array([-np.pi/2,0,0,0.8,0.8,1.1])

    #细化三维血管
    skele_3d_volume = skeletonize_3d(seg_ia_array_dsa3d)
    #获取中心线三维坐标(体素坐标)
    skele_3d_points_voxel = np.argwhere(skele_3d_volume>0)#[N,3]
    skele_3d_points_voxel = np.column_stack((skele_3d_points_voxel,np.ones((len(skele_3d_points_voxel),1)))).T #[4,N]
    #获取中心线三维坐标(图像坐标系)
    skele_3d_points_volume = T_3d_voxel_to_image_dsa3d@skele_3d_points_voxel

    """读取cor影像"""
    #读取影像数据
    dsa2d_seg_cor_path = data_dir+"/seg_cor.nii.gz"
    seg_ia_img_dsa2d_cor, seg_ia_array_dsa2d_cor = read_dsa_2d_seg(dsa2d_seg_cor_path)
    #细化二维血管
    skele_2d_cor_image = skeletonize(seg_ia_array_dsa2d_cor)
    #获取中心线二维坐标
    skele_2d_cor_points_pixel = np.argwhere(skele_2d_cor_image>0).T


    """读取sag影像"""
    #读取影像数据
    dsa2d_seg_sag_path = data_dir+"/seg_sag.nii"
    seg_ia_img_dsa2d_sag, seg_ia_array_dsa2d_sag = read_dsa_2d_seg(dsa2d_seg_sag_path)
    #细化二维血管
    skele_2d_sag_image = skeletonize(seg_ia_array_dsa2d_sag)
    #获取中心线二维坐标
    skele_2d_sag_points_pixel = np.argwhere(skele_2d_sag_image>0)[:,0:2].T


    """读取投影矩阵"""
    with open(os.path.join(data_dir,"parameters.json"),"r") as file:
        param_json = json.load(file)
    #正位
    P_cor = np.array(param_json["P_cor"])
    dsa_param_cor = param_json["dsa_param_cor"]
    #侧位
    P_sag = np.array(param_json["P_sag"])
    dsa_param_sag = param_json["dsa_param_sag"]

    """对三维点集与二维点集进行配准"""
    proj_points_cor = P_cor@skele_3d_points_volume
    s1_p_sag = proj_points_cor/np.expand_dims(proj_points_cor[2],axis=0)
    near_id_cor = similarity(proj_points_cor[0:2],skele_2d_cor_points_pixel[0:2])
    skele_2d_cor_points_pixel_near = skele_2d_cor_points_pixel[:,near_id_cor]
    
    proj_points_sag = P_sag@skele_3d_points_volume
    s1_p_sag = proj_points_sag/np.expand_dims(proj_points_sag[2],axis=0)
    near_id_sag = similarity(proj_points_sag[0:2],skele_2d_sag_points_pixel[0:2])
    skele_2d_sag_points_pixel_near = skele_2d_sag_points_pixel[:,near_id_sag]

    """在三维点集中采样控制点集"""
    control_points = skele_3d_points_volume[0:3,::4]
    sigma = 3*np.ones((control_points.shape[1],1))
    lambda_norm = 1


    """进行优化更新T"""
    weight = np.random.rand(3*control_points.shape[1])
    problem = optimize.OptimizeResult(
    fun = Tikhonov_regularization_two_view,
    x0 = weight,
    args = (P_cor,
            P_sag,
            control_points,
            skele_3d_points_volume[0:3],
            skele_2d_cor_points_pixel_near[0:2],
            skele_2d_sag_points_pixel_near[0:2],
            sigma,
            lambda_norm),
    method = "BFGS",
            # bounds = [(-10000,10000),(-10000,10000),(-10000,10000),
            #           (-500,500),(-500,500),(-500,500)]

        )
    optim_result = optimize.minimize(**problem)
    weight = optim_result.x