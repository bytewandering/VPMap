import SimpleITK as sitk
import numpy as np
import torch
from matplotlib import pyplot as plt
import os
import time
import json
from collections import OrderedDict
from matplotlib.colors import ListedColormap
import cv2

from utils.read import read_dsa_2d_ori,read_dsa_2d_seg
from utils.acc import dice, mean_skeleton_distance
from utils.cip import CIP
from utils.show import show

from methods.ICP.icp import ICP_for_2d_3d_registration
from methods.ICB.icp_hms import ICP_HMS
from methods.ICP_with_adaptive_error.icp_with_adaptive_error import ICP_PWO
from methods.ICB.icb import PPHReg
from methods.PPHReg_for_joint.morpho_register import adaptive_ICP_for_2d_3d_registration_v2
from methods.ICB.icb_plus import ICB

from utils.transform_matrix import w_to_T, camera_matrix, image_plane_matrix


#世界坐标系：以放射源到接收器中心的连线和patient plane之间的交点为原点，垂直向上为z轴正方向，LAO为x轴正方向，CRA为y轴正方向
#相机坐标系：放射源为原点，放射源到图像中心的射线方向为z轴正方向，x轴y轴方向与像素坐标系的x轴y轴方向保持一致
#图像坐标系：图像中心为原点，向右为x轴正方向，向下为y轴正方向
#像素坐标系：图像左上角为原点，向右为x轴正方向，向下为y轴正方向




if __name__ == "__main__":

    orig_dir_path = "F:/3D_2D/VAPPF/test_data"
    #输入的分割图来源
    test_dir_path = "F:/3D_2D/VAPPF/pred_data/VMNet/predictions/test"
    #输入的分割图的名称
    test_name = "seg.png"
    #输出的配准信息
    out_put_dir = "F:/3D_2D/VAPPF_info/VMNet+PPHReg"
    
    dir_list = os.listdir(orig_dir_path)
    w_list = []
    rotation_error_count = []
    translation_error_count = []
    dist_cor_count = []
    dice_cor_count = []
    dist_sag_count = []
    dice_sag_count = []

    process_time_count = []

    for i in range(0,len(dir_list)):
        orig_data_dir = orig_dir_path + "/" + dir_list[i]
        test_data_dir = test_dir_path + "/" + dir_list[i]

        print("registration ",dir_list[i])
        """读取三维影像"""
        #读取影像数据
        dsa3d_seg_path = orig_data_dir+"/3D/3D_DSA_seg.nii.gz"
        seg_ia_img_dsa3d = sitk.ReadImage(dsa3d_seg_path)
        seg_ia_array_dsa3d = sitk.GetArrayFromImage(seg_ia_img_dsa3d).transpose(2,1,0)
        space_3d = seg_ia_img_dsa3d.GetSpacing()
        size_3d = seg_ia_img_dsa3d.GetSize()

        #生成T_3d_voxel_to_img
        T_3d_voxel_to_image_dsa3d = np.array([[space_3d[0],0          ,0          ,-(size_3d[0]-1)*space_3d[0]/2],
                                              [0          ,space_3d[1],0          ,-(size_3d[1]-1)*space_3d[1]/2],
                                              [0          ,0          ,space_3d[2],-(size_3d[2]-1)*space_3d[2]/2],
                                              [0          ,0          ,0          ,1                            ]])

        """修改配准方法需要修改的地方1/3:1,2,5,8,9"""
        w0 = np.array([-np.pi/2,np.pi/4,0,10,10,0])

        # w0 = np.array([-np.pi/2,np.pi/4,0,10,10,10])

        # # w0 = np.array([-np.pi/2,np.pi/4,np.pi/4,10,10,10])
        # # w0 = np.array([-np.pi/4,np.pi/4,np.pi/4,10,10,10])

        # w0 = np.array([-np.pi/2,np.pi/2,0,10,10,10])

        # # w0 = np.array([-np.pi/2,np.pi/2,np.pi/2,10,10,10])
        # # w0 = np.array([0,np.pi/2,np.pi/2,10,10,10])

        # w0 = np.array([-np.pi/2,np.pi*3/4,0,10,10,10])

        # w0 = np.array([-np.pi/2,np.pi,0,10,10,10])

        """读取cor影像"""
        #读取影像数据
        dsa2d_seg_cor_path = test_data_dir+"/cor/"+test_name
        seg_ia_array_dsa2d_cor = cv2.imread(dsa2d_seg_cor_path,cv2.IMREAD_GRAYSCALE)
        seg_ia_array_dsa2d_cor = cv2.resize(seg_ia_array_dsa2d_cor,(750,750),interpolation=cv2.INTER_NEAREST)
        seg_ia_array_dsa2d_cor = np.expand_dims(np.where(seg_ia_array_dsa2d_cor>0,1,0),axis=-1)
        seg_ia_array_dsa2d_cor = seg_ia_array_dsa2d_cor.transpose([1,0,2])
        # seg_ia_array_dsa2d_cor = read_dsa_2d_seg(dsa2d_seg_cor_path)
        # cmap1 = ListedColormap(["white","blue"])
        # plt.imshow(seg_ia_array_dsa2d_cor,
        #             cmap=cmap1, 
        #             alpha=1.0, 
        #             vmin=np.min(seg_ia_array_dsa2d_cor), 
        #             vmax=np.max(seg_ia_array_dsa2d_cor))
        # plt.show()
        

        """读取sag影像"""
        #读取影像数据
        dsa2d_seg_sag_path = test_data_dir+"/sag/"+test_name
        seg_ia_array_dsa2d_sag = cv2.imread(dsa2d_seg_sag_path,cv2.IMREAD_GRAYSCALE)
        seg_ia_array_dsa2d_sag = cv2.resize(seg_ia_array_dsa2d_sag,(750,750),interpolation=cv2.INTER_NEAREST)
        seg_ia_array_dsa2d_sag = np.expand_dims(np.where(seg_ia_array_dsa2d_sag>0,1,0),axis=-1)
        seg_ia_array_dsa2d_sag = seg_ia_array_dsa2d_sag.transpose([1,0,2])
        # seg_ia_img_dsa2d_sag, seg_ia_array_dsa2d_sag = read_dsa_2d_seg(dsa2d_seg_sag_path)



        """读取投影矩阵"""
        with open(os.path.join(orig_data_dir,"parameters.json"),"r") as file:
            param_json = json.load(file)
        #正位
        P_cor = np.array(param_json["P_cor"])
        dsa_param_cor = param_json["dsa_param_cor"]
        #侧位
        P_sag = np.array(param_json["P_sag"])
        dsa_param_sag = param_json["dsa_param_sag"]

        """读取真实位姿"""
        w_ground_truth = np.array(param_json["w_ground_truth"])


        """配准"""
        """修改配准方法需要修改的地方2/3"""
        start_time = time.time()

        """ICP"""
        # w,T,obj_err_cor,obj_err_sag = ICP_for_2d_3d_registration(seg_ia_array_dsa3d,
        #                                                          seg_ia_array_dsa2d_cor,
        #                                                          seg_ia_array_dsa2d_sag,
        #                                                          T_3d_voxel_to_image_dsa3d,
        #                                                          w0,
        #                                                          P_cor,
        #                                                          P_sag)

        """ICP+HMS"""
        # w,T,obj_err_cor,obj_err_sag = ICP_HMS(seg_ia_array_dsa3d,
        #                                      seg_ia_array_dsa2d_cor,
        #                                      seg_ia_array_dsa2d_sag,
        #                                     T_3d_voxel_to_image_dsa3d,
        #                                     w0,
        #                                     P_cor,
        #                                     dsa_param_cor[2:],
        #                                     P_sag,
        #                                     dsa_param_sag[2:])

        """ICP+PWO"""
        w,T,obj_err_cor,obj_err_sag = ICP_PWO(seg_ia_array_dsa3d,
                                             seg_ia_array_dsa2d_cor,
                                             seg_ia_array_dsa2d_sag,
                                            T_3d_voxel_to_image_dsa3d,
                                            w0,
                                            P_cor,
                                            dsa_param_cor[2:],
                                            P_sag,
                                            dsa_param_sag[2:])
    
        """PPHReg"""
        # w,T,obj_err_cor,obj_err_sag = PPHReg(seg_ia_array_dsa3d,
        #                                      seg_ia_array_dsa2d_cor,
        #                                      seg_ia_array_dsa2d_sag,
        #                                      T_3d_voxel_to_image_dsa3d,
        #                                      w0,
        #                                      P_cor,
        #                                      dsa_param_cor[2:],
        #                                      P_sag,
        #                                      dsa_param_sag[2:])

        """PPHReg_for_joint"""
        # w,T,obj_err_cor,obj_err_sag = adaptive_ICP_for_2d_3d_registration_v2(seg_ia_array_dsa3d,
        #                                                                   seg_ia_array_dsa2d_cor,
        #                                                                   seg_ia_array_dsa2d_sag,
        #                                                                   T_3d_voxel_to_image_dsa3d,
        #                                                                   w0,
        #                                                                   P_cor,
        #                                                                   dsa_param_cor[2:],
        #                                                                   P_sag,
        #                                                                   dsa_param_sag[2:])

        """废弃版本"""
        # w,T,obj_err_cor,obj_err_sag = ICB(seg_ia_array_dsa3d,
        #                                                                   seg_ia_array_dsa2d_cor,
        #                                                                   seg_ia_array_dsa2d_sag,
        #                                                                   T_3d_voxel_to_image_dsa3d,
        #                                                                   w0,
        #                                                                   P_cor,
        #                                                                   dsa_param_cor[2:],
        #                                                                   P_sag,
        #                                                                   dsa_param_sag[2:])        

        process_time = time.time()-start_time
        print("process time:", process_time)
        process_time_count.append(process_time)



        # """累计损失"""
        # #计算位姿误差
        # for angle_i in range(0,3):
        #     if w[angle_i] > 2*np.pi:
        #         w[angle_i] = w[angle_i]%(2*np.pi)
        #     elif w[angle_i] < -2*np.pi:
        #         w[angle_i] = w[angle_i] - np.ceil(w[angle_i]/2/np.pi)*2*np.pi
        # w_error = (w - w_ground_truth)**2
        # rotation_error = np.sqrt(np.sum(w_error[0:3]))*180/np.pi
        # rotation_error = rotation_error if 360-rotation_error>rotation_error else 360-rotation_error
        # rotation_error_count.append(rotation_error)
        # translation_error = np.sqrt(np.sum(w_error[3:]))
        # translation_error_count.append(translation_error)

        # print("rotation_error:",rotation_error)
        # print("translation_error:",translation_error)

        # #冠状面上中心线投影距离
        # pd_cor = mean_skeleton_distance(seg_ia_array_dsa3d,
        #                                 T_3d_voxel_to_image_dsa3d,
        #                                 w,
        #                                 seg_ia_array_dsa2d_cor,
        #                                 P_cor)
        # print("PD cor:", pd_cor)
        # dist_cor_count.append(pd_cor)

        # #计算冠状面上的投影的dice损失
        # seg_ia_array_dsa2d_cor_proj = CIP(seg_ia_array_dsa3d,
        #                                   T_3d_voxel_to_image_dsa3d,
        #                                   T,
        #                                   P_cor,
        #                                   seg_ia_array_dsa2d_cor.shape[0:2]
        #                                   )
        


        # dice_cor = dice(torch.from_numpy(seg_ia_array_dsa2d_cor_proj).unsqueeze(0).unsqueeze(0), 
        #                 torch.from_numpy(seg_ia_array_dsa2d_cor.sum(-1)).unsqueeze(0).unsqueeze(0))
        # print("dice cor:", dice_cor)
        # dice_cor_count.append(dice_cor)

        # #计算矢状面上中心线投影距离
        # pd_sag = mean_skeleton_distance(seg_ia_array_dsa3d,
        #                        T_3d_voxel_to_image_dsa3d,
        #                        w,
        #                        seg_ia_array_dsa2d_sag,
        #                        P_sag)
        # print("PD sag:", pd_sag)
        # dist_sag_count.append(pd_sag)

        # #计算矢状面上的投影的dice损失
        # seg_ia_array_dsa2d_sag_proj = CIP(seg_ia_array_dsa3d,
        #                                   T_3d_voxel_to_image_dsa3d,
        #                                   T,
        #                                   P_sag,
        #                                   seg_ia_array_dsa2d_sag.shape[0:2]
        #                                   )
        # dice_sag = dice(torch.from_numpy(seg_ia_array_dsa2d_sag_proj).unsqueeze(0).unsqueeze(0), 
        #                 torch.from_numpy(seg_ia_array_dsa2d_sag.sum(-1)).unsqueeze(0).unsqueeze(0))
        # print("dice sag:", dice_sag)
        # dice_sag_count.append(dice_sag)


        """输出投影图像"""
        """修改配准方法需要修改的地方3/3"""
        out_put_path = out_put_dir + "/" + dir_list[i]
        if not os.path.exists(out_put_path):
            os.makedirs(out_put_path)

        # # 输出配准前的中心线点
        # show(P_cor,
        #      T_3d_voxel_to_image_dsa3d,
        #      w0,
        #      seg_ia_array_dsa3d,
        #      seg_ia_array_dsa2d_cor,
        #      out_put_path+"/before_registration_cor.png")
        
        # show(P_sag,
        #      T_3d_voxel_to_image_dsa3d,
        #      w0,
        #      seg_ia_array_dsa3d,
        #      seg_ia_array_dsa2d_sag,
        #      out_put_path+"/before_registration_sag.png")

        # #输出配准后的中心线点
        # show(P_cor,
        #      T_3d_voxel_to_image_dsa3d,
        #      w,
        #      seg_ia_array_dsa3d,
        #      seg_ia_array_dsa2d_cor,
        #      out_put_path+"/after_registration_cor.png")
        
        # show(P_sag,
        #      T_3d_voxel_to_image_dsa3d,
        #      w,
        #      seg_ia_array_dsa3d,
        #      seg_ia_array_dsa2d_sag,
        #      out_put_path+"/after_registration_sag.png")
        
        # #输出冠状面投影图像
        # seg_ia_image_dsa2d_cor_proj = sitk.GetImageFromArray(np.expand_dims(seg_ia_array_dsa2d_cor_proj,-1).repeat(seg_ia_array_dsa2d_cor.shape[-1],axis=-1).transpose(2,1,0))
        # seg_ia_image_dsa2d_cor_proj.CopyInformation(seg_ia_img_dsa2d_cor)
        # sitk.WriteImage(seg_ia_image_dsa2d_cor_proj, out_put_path+"/seg_cor_proj.nii.gz")

        # #输出矢状面投影图像
        # seg_ia_image_dsa2d_sag_proj = sitk.GetImageFromArray(np.expand_dims(seg_ia_array_dsa2d_sag_proj,-1).repeat(seg_ia_array_dsa2d_sag.shape[-1],axis=-1).transpose(2,1,0))
        # seg_ia_image_dsa2d_sag_proj.CopyInformation(seg_ia_img_dsa2d_sag)
        # sitk.WriteImage(seg_ia_image_dsa2d_sag_proj, out_put_path+"/seg_sag_proj.nii.gz")

        single_json = OrderedDict()
        single_json["w_init"] = w0.tolist()
        single_json["w_ground_truth"] = w_ground_truth.tolist()
        single_json["w_registration"] = w.tolist()
        single_json["process_time"] = process_time


        with open(out_put_path+"/result.json","w") as f:
            json.dump(single_json,f,indent=4)
            print("json文件生成成功")


    # """计算准确率的均值"""
    # process_time_count = np.array(process_time_count).astype(np.float64)
    # process_time_mean = process_time_count.mean()
    # process_time_std = process_time_count.std()

    # rotation_error_count = np.array(rotation_error_count).astype(np.float64)
    # rotation_error_mean = rotation_error_count.mean()
    # rotation_error_std = rotation_error_count.std()

    # translation_error_count = np.array(translation_error_count).astype(np.float64)
    # translation_error_mean = translation_error_count.mean()
    # translation_error_std = translation_error_count.std()

    # dist_cor_count = (np.array(dist_cor_count)*0.4).astype(np.float64)
    # dist_cor_mean = dist_cor_count.mean()
    # dist_cor_std = dist_cor_count.std()

    # dice_cor_count = np.array(dice_cor_count).astype(np.float64)
    # dice_cor_mean = dice_cor_count.mean()
    # dice_cor_std = dice_cor_count.std()

    # dist_sag_count = (np.array(dist_sag_count)*0.4).astype(np.float64)
    # dist_sag_mean = dist_sag_count.mean()
    # dist_sag_std = dist_sag_count.std()

    # dice_sag_count = np.array(dice_sag_count).astype(np.float64)
    # dice_sag_mean = dice_sag_count.mean()
    # dice_sag_std = dice_sag_count.std()

    # """计算成功率"""
    # success_ratio_1 = np.mean(np.where(translation_error_count<1,1,0)&np.where(rotation_error_count<1,1,0))
    # success_ratio_5 = np.mean(np.where(translation_error_count<5,1,0)&np.where(rotation_error_count<5,1,0))
    # success_ratio_10 = np.mean(np.where(translation_error_count<10,1,0)&np.where(rotation_error_count<10,1,0))
    # success_ratio_15 = np.mean(np.where(translation_error_count<15,1,0)&np.where(rotation_error_count<15,1,0))
    # success_ratio_20 = np.mean(np.where(translation_error_count<20,1,0)&np.where(rotation_error_count<20,1,0))
    # success_ratio_25 = np.mean(np.where(translation_error_count<25,1,0)&np.where(rotation_error_count<25,1,0))


    # json_dict = OrderedDict()
    # json_dict["process_time(s) mean(std)"] = [process_time_mean, process_time_std]
    
    # json_dict["translation_error(mm) mean(std)"] = [translation_error_mean, translation_error_std]
    # json_dict["rotation_error(degree) mean(std)"] = [rotation_error_mean, rotation_error_std]

    # json_dict["PD_cor(mm) mean(std)"] = [dist_cor_mean, dist_cor_std]
    # json_dict["PD_sag(mm) mean(std)"] = [dist_sag_mean, dist_sag_std]
    
    # json_dict["dice_cor(%) mean(std)"] = [dice_cor_mean, dice_cor_std]
    # json_dict["dice_sag(%) mean(std)"] = [dice_sag_mean, dice_sag_std]

    # json_dict["success_ratio_1"] = success_ratio_1
    # json_dict["success_ratio_5"] = success_ratio_5
    # json_dict["success_ratio_10"] = success_ratio_10
    # json_dict["success_ratio_15"] = success_ratio_15
    # json_dict["success_ratio_20"] = success_ratio_20
    # json_dict["success_ratio_25"] = success_ratio_25

    # with open(out_put_dir+"/results.json","w") as f:
    #     json.dump(json_dict,f,indent=4)
    #     print("json文件生成成功")
    
    # print("process_time(s): mean-{}, std-{}".format(process_time_mean, process_time_std))

    # print("translation_error(mm): mean-{}, std-{}".format(translation_error_mean, translation_error_std))
    # print("rotation_error(degree): mean-{}, std-{}".format(rotation_error_mean, rotation_error_std))

    # print("dice_cor(%): mean-{}, std-{}".format(dice_cor_mean, dice_cor_std))
    # print("dice_sag(%): mean-{}, std-{}".format(dice_sag_mean, dice_sag_std))

    # print("PD_cor(mm): mean-{}, std-{}".format(dist_cor_mean, dist_cor_std))
    # print("PD_sag(mm): mean-{}, std-{}".format(dist_sag_mean, dist_sag_std))

    # print("success_ratio_1: {}".format(success_ratio_1))
    # print("success_ratio_5: {}".format(success_ratio_5))
    # print("success_ratio_10: {}".format(success_ratio_10))
    # print("success_ratio_15: {}".format(success_ratio_15))
    # print("success_ratio_20: {}".format(success_ratio_20))
    # print("success_ratio_25: {}".format(success_ratio_25))



