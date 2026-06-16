import numpy as np
import SimpleITK as sitk
from .transform_matrix import w_to_T, camera_matrix, image_plane_matrix

def read_dsa_2d_ori(origin_path):
    #读取冠状面二维原始图
    dsa_ori_2d_img = sitk.ReadImage(origin_path)
    dsa_ori_2d_array = sitk.GetArrayFromImage(dsa_ori_2d_img).transpose(2,1,0)


    #计算参数
    intrinsic, extrinsic, d, f, angle1, angle2 = camera_matrix(dsa_ori_2d_img)
    # return dsa_ori_2d_img,dsa_ori_2d_array,d,f,angle1,angle2
    P = intrinsic@extrinsic
    image_plane = image_plane_matrix(dsa_ori_2d_img)

    return dsa_ori_2d_img,dsa_ori_2d_array,P,image_plane,[d,f,angle1,angle2]

def read_dsa_2d_seg(seg_path):
    
    #读取冠状面二维分割图
    dsa_seg_2d_img = sitk.ReadImage(seg_path)
    dsa_seg_2d_array = sitk.GetArrayFromImage(dsa_seg_2d_img).transpose(2,1,0)
    dsa_seg_2d_array = np.ascontiguousarray(dsa_seg_2d_array)

    return dsa_seg_2d_img, dsa_seg_2d_array