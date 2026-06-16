import os
import numpy as np
import SimpleITK as sitk
from collections import OrderedDict
import json
from transform_matrix import w_to_T, camera_matrix, image_plane_matrix



def read_dsa_2d_ori(origin_path):
    #读取冠状面二维原始图
    dsa_ori_2d_img = sitk.ReadImage(origin_path)
    dsa_ori_2d_array = sitk.GetArrayFromImage(dsa_ori_2d_img).transpose(2,1,0)


    #计算参数
    intrinsic, extrinsic, d, f, angle1, angle2 = camera_matrix(dsa_ori_2d_img)
    # return dsa_ori_2d_img,dsa_ori_2d_array,d,f,angle1,angle2
    P = intrinsic@extrinsic
    image_plane = image_plane_matrix(dsa_ori_2d_img)

    return dsa_ori_2d_img,dsa_ori_2d_array,P,d,f,angle1,angle2


    
if __name__=="__main__":
    input_dir = "F:/3D_2D/original_data"
    output_dir = "F:/3D_2D/VAPPF/test_data"

    data_list = os.listdir(input_dir)

    for i in range(len(data_list)):
        data_name = data_list[i]

        cor_path = input_dir+"/"+data_name+"/2D_DSA_cor.dcm"
        img_cor,array_cor,P_cor,d_cor,f_cor,angle1_cor,angle2_cor = read_dsa_2d_ori(cor_path)
        spacing_cor = img_cor.GetSpacing()


        sag_path = input_dir+"/"+data_name+"/2D_DSA_sag.dcm"
        img_sag,array_sag,P_sag,d_sag,f_sag,angle1_sag,angle2_sag = read_dsa_2d_ori(sag_path)
        spacing_sag = img_sag.GetSpacing()
        
        parameter_dict = OrderedDict()
        parameter_dict["w_ground_truth"] = [-np.pi/2,0,0,
                                            0,1,-1.5]
        parameter_dict["P_cor"] = P_cor.tolist()
        parameter_dict["P_sag"] = P_sag.tolist()
        parameter_dict["dsa_param_cor"] = [d_cor,f_cor,angle1_cor,angle2_cor] 
        parameter_dict["dsa_param_sag"] = [d_sag,f_sag,angle1_sag,angle2_sag] 
        parameter_dict["space_cor"] = spacing_cor
        parameter_dict["space_sag"] = spacing_sag

        ouput_path = output_dir+"/"+data_name
        with open(ouput_path+"/parameters.json","w") as f:
            json.dump(parameter_dict,f,indent=4)
            print("json文件生成成功")



