import SimpleITK as sitk
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import os
import time
import json
from collections import OrderedDict

from skimage.morphology import skeletonize,skeletonize_3d
from utils.read import read_dsa_2d_ori,read_dsa_2d_seg
from utils.acc import dice, mean_skeleton_distance,mean_target_registration_distance,mean_projection_distance
from utils.cip import CIP
from utils.show import show
from utils.transform_matrix import w_to_T

from methods.ICP.icp import ICP_for_2d_3d_registration
from methods.ICP_with_adaptive_error.icp_with_adaptive_error import ICP_PWO
from methods.ICB.icb import PPHReg
from methods.ICB.icb_plus import ICB




#世界坐标系：以放射源到接收器中心的连线和patient plane之间的交点为原点，垂直向上为z轴正方向，LAO为x轴正方向，CRA为y轴正方向
#相机坐标系：放射源为原点，放射源到图像中心的射线方向为z轴正方向，x轴y轴方向与像素坐标系的x轴y轴方向保持一致
#图像坐标系：图像中心为原点，向右为x轴正方向，向下为y轴正方向
#像素坐标系：图像左上角为原点，向右为x轴正方向，向下为y轴正方向



if __name__ == "__main__":

    pred_dir_path = "F:/backup/data/IAS_train_info/IAS_2D_3D_Registration/norm_data/PPHReg/9"
    image_dir_path = "F:/backup/data/IAS/IAS_2D_3D_Registration/using"

    dir_list = os.listdir(image_dir_path)

    w_list = []
    rotation_error_count = []
    translation_error_count = []
    mTRE_count = []
    PD_cor_count = []
    PD_sag_count = []
    dice_cor_count = []
    dice_sag_count = []

    process_time_count = []

    index_list = []

    # for i in range(0,len(dir_list)):
    for i in range(4,8):

        image_dir = image_dir_path + "\\" +dir_list[i]
        pred_dir = pred_dir_path + "\\" +dir_list[i]

        print("########registration ",dir_list[i],"########")
        
        """读取三维影像"""
        #读取影像数据
        dsa3d_seg_path = image_dir+"\\seg_3d.nii.gz"
        seg_ia_img_dsa3d = sitk.ReadImage(dsa3d_seg_path)
        seg_ia_array_dsa3d = sitk.GetArrayFromImage(seg_ia_img_dsa3d).transpose(2,1,0)
        space_3d = seg_ia_img_dsa3d.GetSpacing()
        size_3d = seg_ia_img_dsa3d.GetSize()

        #生成T_3d_voxel_to_img
        T_3d_voxel_to_image_dsa3d = np.array([[space_3d[0],0          ,0          ,-(size_3d[0]-1)*space_3d[0]/2],
                                              [0          ,space_3d[1],0          ,-(size_3d[1]-1)*space_3d[1]/2],
                                              [0          ,0          ,space_3d[2],-(size_3d[2]-1)*space_3d[2]/2],
                                              [0          ,0          ,0          ,1                            ]])
        
        #细化三维血管
        skele_3d_volume = skeletonize_3d(seg_ia_array_dsa3d)
        #获取中心线三维坐标(体素坐标)
        skele_3d_points_voxel = np.argwhere(skele_3d_volume>0)#[N,3]
        skele_3d_points_voxel = np.column_stack((skele_3d_points_voxel,np.ones((len(skele_3d_points_voxel),1)))).T #[4,N]


        """读取cor影像"""
        #读取影像数据
        dsa2d_seg_cor_path = image_dir+"\\seg_cor.nii.gz"
        seg_ia_img_dsa2d_cor, seg_ia_array_dsa2d_cor = read_dsa_2d_seg(dsa2d_seg_cor_path)


        #细化二维血管
        skele_2d_image_cor = skeletonize(seg_ia_array_dsa2d_cor)
        #获取中心线二维坐标
        skele_2d_points_pixel_cor = np.argwhere(skele_2d_image_cor>0)[:,0:2]
        skele_2d_points_pixel_cor = np.column_stack((skele_2d_points_pixel_cor,np.ones((len(skele_2d_points_pixel_cor),1)))).T #[3,N]
        

        """读取sag影像"""
        #读取影像数据
        dsa2d_seg_sag_path = image_dir+"\\seg_sag.nii.gz"
        seg_ia_img_dsa2d_sag, seg_ia_array_dsa2d_sag = read_dsa_2d_seg(dsa2d_seg_sag_path)

        #细化二维血管
        skele_2d_image_sag = skeletonize(seg_ia_array_dsa2d_sag)
        #获取中心线二维坐标
        skele_2d_points_pixel_sag = np.argwhere(skele_2d_image_sag>0)[:,0:2]
        skele_2d_points_pixel_sag = np.column_stack((skele_2d_points_pixel_sag,np.ones((len(skele_2d_points_pixel_sag),1)))).T #[3,N]


        """读取投影矩阵"""
        with open(os.path.join(image_dir,"parameters.json"),"r") as file:
            param_json = json.load(file)
        #正位
        P_cor = np.array(param_json["P_cor"])
        dsa_param_cor = param_json["dsa_param_cor"]
        #侧位
        P_sag = np.array(param_json["P_sag"])
        dsa_param_sag = param_json["dsa_param_sag"]


        """读取预测的json文件"""
        with open(os.path.join(pred_dir,"result.json"),"r") as file:
            pred_json = json.load(file)

        w0 = np.array(pred_json["w_init"])
        T0 = w_to_T(w0)
        w_ground_truth = np.array(pred_json["w_ground_truth"])
        T_ground_truth = w_to_T(w_ground_truth)
        w_pred = np.array(pred_json["w_registration"])
        T_pred = w_to_T(w_pred)





        """计算位移误差TE"""
        w_error = (w_pred[3:] - w_ground_truth[3:])**2
        translation_error = np.sqrt(np.sum(w_error))
        print("translation_error:",translation_error)
        translation_error_count.append(translation_error)

        """计算角度误差RE"""
        rot_ground_truth = T_ground_truth[0:3,0:3]
        rot_pred = T_pred[0:3,0:3]
        rot_error = (rot_pred-rot_ground_truth)**2
        rotation_error = np.sqrt(np.sum(rot_error))
        print("rotation_error:",rotation_error)
        rotation_error_count.append(rotation_error)
        

        """计算mean target registration error(mTRE)"""
        mTRE = mean_target_registration_distance(skele_3d_points_voxel,
                                                 T_3d_voxel_to_image_dsa3d,
                                                 w_ground_truth,
                                                 w_pred)
        print("mTRE:",mTRE)
        mTRE_count.append(mTRE)

        """计算正位投影距离PD_a"""
        PD_cor = mean_projection_distance(skele_3d_points_voxel,
                                                 T_3d_voxel_to_image_dsa3d,
                                                 w_ground_truth,
                                                 w_pred,
                                                 P_cor)
        print("PD_cor:",PD_cor)
        PD_cor_count.append(PD_cor)

        """计算侧位投影距离PD_l"""
        PD_sag = mean_projection_distance(skele_3d_points_voxel,
                                                 T_3d_voxel_to_image_dsa3d,
                                                 w_ground_truth,
                                                 w_pred,
                                                 P_sag)
        print("PD_sag:",PD_sag)
        PD_sag_count.append(PD_sag)

        """计算正位投影的dice损失"""
        seg_ia_array_dsa2d_cor_proj = CIP(seg_ia_array_dsa3d,
                                          T_3d_voxel_to_image_dsa3d,
                                          T_pred,
                                          P_cor,
                                          seg_ia_array_dsa2d_cor.shape[0:2]
                                          )
        
        dice_cor = dice(torch.from_numpy(seg_ia_array_dsa2d_cor_proj).unsqueeze(0).unsqueeze(0), 
                        torch.from_numpy(seg_ia_array_dsa2d_cor.sum(-1)).unsqueeze(0).unsqueeze(0))
        print("dice cor:", dice_cor)
        dice_cor_count.append(dice_cor)


        """计算侧位投影的dice损失"""
        seg_ia_array_dsa2d_sag_proj = CIP(seg_ia_array_dsa3d,
                                          T_3d_voxel_to_image_dsa3d,
                                          T_pred,
                                          P_sag,
                                          seg_ia_array_dsa2d_sag.shape[0:2]
                                          )
        dice_sag = dice(torch.from_numpy(seg_ia_array_dsa2d_sag_proj).unsqueeze(0).unsqueeze(0), 
                        torch.from_numpy(seg_ia_array_dsa2d_sag.sum(-1)).unsqueeze(0).unsqueeze(0))
        print("dice sag:", dice_sag)
        dice_sag_count.append(dice_sag)



        single_eval_json = OrderedDict()
        single_eval_json["translation_error(mm)"] = translation_error
        single_eval_json["rotation_error(degree)"] = rotation_error
        single_eval_json["mTRE(mm)"] = mTRE
        single_eval_json["PD_cor(mm)"] = PD_cor
        single_eval_json["PD_sag(mm)"] = PD_sag
        single_eval_json["dice_cor(%)"] = dice_cor.tolist()
        single_eval_json["dice_sag(%)"] = dice_sag.tolist()


        with open(pred_dir+"/eval.json","w") as f:
            json.dump(single_eval_json,f,indent=4)
            print("json文件生成成功")


        

        # 输出配准前的中心线点
        show(P_cor,
             T_3d_voxel_to_image_dsa3d,
             w0,
             seg_ia_array_dsa2d_cor,
             skele_3d_points_voxel,
             skele_2d_points_pixel_cor,
             pred_dir+"/before_registration_cor.png")
        
        show(P_sag,
             T_3d_voxel_to_image_dsa3d,
             w0,
             seg_ia_array_dsa2d_sag,
             skele_3d_points_voxel,
             skele_2d_points_pixel_sag,
             pred_dir+"/before_registration_sag.png")

        #输出配准后的中心线点
        show(P_cor,
             T_3d_voxel_to_image_dsa3d,
             w_pred,
             seg_ia_array_dsa2d_cor,
             skele_3d_points_voxel,
             skele_2d_points_pixel_cor,
             pred_dir+"/after_registration_cor.png")
        
        show(P_sag,
             T_3d_voxel_to_image_dsa3d,
             w_pred,
             seg_ia_array_dsa2d_sag,
             skele_3d_points_voxel,
             skele_2d_points_pixel_sag,
             pred_dir+"/after_registration_sag.png")


        """配准前冠状面投影"""
        seg_ia_array_dsa2d_cor_proj_before_reg = CIP(seg_ia_array_dsa3d,
                                                    T_3d_voxel_to_image_dsa3d,
                                                    T0,
                                                    P_cor,
                                                    seg_ia_array_dsa2d_cor.shape[0:2]
                                                    )

        seg_ia_array_dsa2d_cor_proj_before_reg = np.expand_dims(seg_ia_array_dsa2d_cor_proj_before_reg,axis=-1)
        plt.figure()
        plt.tight_layout()
        # plt.axis("off")
        cmap1 = ListedColormap(["white","blue"])
        plt.imshow(seg_ia_array_dsa2d_cor.transpose([1,0,2]),
                    cmap=cmap1, 
                    alpha=1.0, 
                    vmin=np.min(seg_ia_array_dsa2d_cor), 
                    vmax=np.max(seg_ia_array_dsa2d_cor))
        cmap2 = ListedColormap(["none","red"])
        plt.imshow(seg_ia_array_dsa2d_cor_proj_before_reg.transpose([1,0,2]),
                    cmap=cmap2, 
                    alpha=0.8, 
                    vmin=np.min(seg_ia_array_dsa2d_cor_proj_before_reg), 
                    vmax=np.max(seg_ia_array_dsa2d_cor_proj_before_reg))
        plt.savefig(pred_dir+"/"+"/proj_cor_before_reg.png", dpi=512, bbox_inches='tight')
        plt.close()

        """配准前冠状面投影"""
        seg_ia_array_dsa2d_cor_proj_after_reg = CIP(seg_ia_array_dsa3d,
                                                    T_3d_voxel_to_image_dsa3d,
                                                    T_pred,
                                                    P_cor,
                                                    seg_ia_array_dsa2d_cor.shape[0:2]
                                                    )

        seg_ia_array_dsa2d_cor_proj_after_reg = np.expand_dims(seg_ia_array_dsa2d_cor_proj_after_reg,axis=-1)
        plt.figure()
        plt.tight_layout()
        # plt.axis("off")
        cmap1 = ListedColormap(["white","blue"])
        plt.imshow(seg_ia_array_dsa2d_cor.transpose([1,0,2]),
                    cmap=cmap1, 
                    alpha=1.0, 
                    vmin=np.min(seg_ia_array_dsa2d_cor), 
                    vmax=np.max(seg_ia_array_dsa2d_cor))
        cmap2 = ListedColormap(["none","red"])
        plt.imshow(seg_ia_array_dsa2d_cor_proj_after_reg.transpose([1,0,2]),
                    cmap=cmap2, 
                    alpha=0.8, 
                    vmin=np.min(seg_ia_array_dsa2d_cor_proj_after_reg), 
                    vmax=np.max(seg_ia_array_dsa2d_cor_proj_after_reg))
        plt.savefig(pred_dir+"/"+"/proj_cor_after_reg.png", dpi=512, bbox_inches='tight')
        plt.close()


        """配准前冠状面投影"""
        seg_ia_array_dsa2d_sag_proj_before_reg = CIP(seg_ia_array_dsa3d,
                                                    T_3d_voxel_to_image_dsa3d,
                                                    T0,
                                                    P_sag,
                                                    seg_ia_array_dsa2d_sag.shape[0:2]
                                                    )

        seg_ia_array_dsa2d_sag_proj_before_reg = np.expand_dims(seg_ia_array_dsa2d_sag_proj_before_reg,axis=-1)
        plt.figure()
        plt.tight_layout()
        # plt.axis("off")
        cmap1 = ListedColormap(["white","blue"])
        plt.imshow(seg_ia_array_dsa2d_sag.transpose([1,0,2]),
                    cmap=cmap1, 
                    alpha=1.0, 
                    vmin=np.min(seg_ia_array_dsa2d_sag), 
                    vmax=np.max(seg_ia_array_dsa2d_sag))
        cmap2 = ListedColormap(["none","red"])
        plt.imshow(seg_ia_array_dsa2d_sag_proj_before_reg.transpose([1,0,2]),
                    cmap=cmap2, 
                    alpha=0.8, 
                    vmin=np.min(seg_ia_array_dsa2d_sag_proj_before_reg), 
                    vmax=np.max(seg_ia_array_dsa2d_sag_proj_before_reg))
        plt.savefig(pred_dir+"/"+"/proj_sag_before_reg.png", dpi=512, bbox_inches='tight')
        plt.close()

        """配准前冠状面投影"""
        seg_ia_array_dsa2d_sag_proj_after_reg = CIP(seg_ia_array_dsa3d,
                                                    T_3d_voxel_to_image_dsa3d,
                                                    T_pred,
                                                    P_sag,
                                                    seg_ia_array_dsa2d_sag.shape[0:2]
                                                    )

        seg_ia_array_dsa2d_sag_proj_after_reg = np.expand_dims(seg_ia_array_dsa2d_sag_proj_after_reg,axis=-1)
        plt.figure()
        plt.tight_layout()
        # plt.axis("off")
        cmap1 = ListedColormap(["white","blue"])
        plt.imshow(seg_ia_array_dsa2d_sag.transpose([1,0,2]),
                    cmap=cmap1, 
                    alpha=1.0, 
                    vmin=np.min(seg_ia_array_dsa2d_sag), 
                    vmax=np.max(seg_ia_array_dsa2d_sag))
        cmap2 = ListedColormap(["none","red"])
        plt.imshow(seg_ia_array_dsa2d_sag_proj_after_reg.transpose([1,0,2]),
                    cmap=cmap2, 
                    alpha=0.8, 
                    vmin=np.min(seg_ia_array_dsa2d_sag_proj_after_reg), 
                    vmax=np.max(seg_ia_array_dsa2d_sag_proj_after_reg))
        plt.savefig(pred_dir+"/"+"/proj_sag_after_reg.png", dpi=512, bbox_inches='tight')
        plt.close()



    # """时间"""
    # process_time_count = np.array(process_time_count).astype(np.float64)
    # process_time_mean = process_time_count.mean()
    # process_time_std = process_time_count.std()

    # """平移误差"""
    # translation_error_count = np.array(translation_error_count).astype(np.float64)
    # translation_error_mean = translation_error_count.mean()
    # translation_error_std = translation_error_count.std()

    # """旋转误差"""
    # rotation_error_count = np.array(rotation_error_count).astype(np.float64)
    # rotation_error_mean = rotation_error_count.mean()
    # rotation_error_std = rotation_error_count.std()

    # """TRE"""
    # mTRE_count = np.array(mTRE_count).astype(np.float64)
    # mTRE_mean = mTRE_count.mean()
    # mTRE_std = mTRE_count.std()

    # """PD_cor"""
    # PD_cor_count = (np.array(PD_cor_count)*0.4).astype(np.float64)
    # PD_cor_mean = PD_cor_count.mean()
    # PD_cor_std = PD_cor_count.std()

    # """PD_sag"""
    # PD_sag_count = (np.array(PD_sag_count)*0.4).astype(np.float64)
    # PD_sag_mean = PD_sag_count.mean()
    # PD_sag_std = PD_sag_count.std()

    # """Dice_cor"""
    # dice_cor_count = np.array(dice_cor_count).astype(np.float64)
    # dice_cor_mean = dice_cor_count.mean()
    # dice_cor_std = dice_cor_count.std()

    # """Dice_sag"""
    # dice_sag_count = np.array(dice_sag_count).astype(np.float64)
    # dice_sag_mean = dice_sag_count.mean()
    # dice_sag_std = dice_sag_count.std()

    # """计算成功率"""
    # success_ratio_1 = np.mean(np.where(mTRE_count<1,1,0))
    # success_ratio_2 = np.mean(np.where(mTRE_count<2,1,0))
    # success_ratio_3 = np.mean(np.where(mTRE_count<3,1,0))
    # success_ratio_4 = np.mean(np.where(mTRE_count<4,1,0))
    # success_ratio_5 = np.mean(np.where(mTRE_count<5,1,0))
    # success_ratio_6 = np.mean(np.where(mTRE_count<6,1,0))
    # success_ratio_7 = np.mean(np.where(mTRE_count<7,1,0))
    # success_ratio_8 = np.mean(np.where(mTRE_count<8,1,0))
    # success_ratio_9 = np.mean(np.where(mTRE_count<9,1,0))
    # success_ratio_10 = np.mean(np.where(mTRE_count<10,1,0))



    # json_dict = OrderedDict()

    # """保存列表数据"""
    # json_dict["translation_error_count"] = translation_error_count.tolist()
    # json_dict["rotation_error_count"] = rotation_error_count.tolist()
    # json_dict["mTRE_count"] = mTRE_count.tolist()
    # json_dict["dice_cor_count"] = dice_cor_count.tolist()
    # json_dict["dice_sag_count"] = dice_sag_count.tolist()
    # json_dict["PD_cor_count"] = PD_cor_count.tolist()
    # json_dict["PD_sag_count"] = PD_sag_count.tolist()

    # """保存均值数据"""
    # json_dict["translation_error(mm) mean(std)"] = [translation_error_mean, translation_error_std]
    # json_dict["rotation_error(degree) mean(std)"] = [rotation_error_mean, rotation_error_std]
    # json_dict["mTRE(mm) mean(std)"] = [mTRE_mean, mTRE_std]
    # json_dict["PD_cor(mm) mean(std)"] = [PD_cor_mean, PD_cor_std]
    # json_dict["PD_sag(mm) mean(std)"] = [PD_sag_mean, PD_sag_std]
    # json_dict["dice_cor(%) mean(std)"] = [dice_cor_mean, dice_cor_std]
    # json_dict["dice_sag(%) mean(std)"] = [dice_sag_mean, dice_sag_std]

    # """保存成功率统计"""
    # json_dict["success_ratio_1"] = success_ratio_1
    # json_dict["success_ratio_2"] = success_ratio_2
    # json_dict["success_ratio_3"] = success_ratio_3
    # json_dict["success_ratio_4"] = success_ratio_4
    # json_dict["success_ratio_5"] = success_ratio_5
    # json_dict["success_ratio_6"] = success_ratio_6
    # json_dict["success_ratio_7"] = success_ratio_7
    # json_dict["success_ratio_8"] = success_ratio_8
    # json_dict["success_ratio_9"] = success_ratio_9
    # json_dict["success_ratio_10"] = success_ratio_10

    # with open(pred_dir_path+"/results.json","w") as f:
    #     json.dump(json_dict,f,indent=4)
    #     print("json文件生成成功")
    
    # print("process_time(s): mean-{}, std-{}".format(process_time_mean, process_time_std))

    # print("translation_error(mm): mean-{}, std-{}".format(translation_error_mean, translation_error_std))
    # print("rotation_error(degree): mean-{}, std-{}".format(rotation_error_mean, rotation_error_std))

    # print("dice_cor(%): mean-{}, std-{}".format(dice_cor_mean, dice_cor_std))
    # print("dice_sag(%): mean-{}, std-{}".format(dice_sag_mean, dice_sag_std))

    # print("PD_cor(mm): mean-{}, std-{}".format(PD_cor_mean, PD_cor_std))
    # print("PD_sag(mm): mean-{}, std-{}".format(PD_sag_mean, PD_sag_std))

    # # print("success_ratio_1: {}".format(success_ratio_1))
    # # print("success_ratio_5: {}".format(success_ratio_5))
    # # print("success_ratio_10: {}".format(success_ratio_10))
    # # print("success_ratio_15: {}".format(success_ratio_15))
    # # print("success_ratio_20: {}".format(success_ratio_20))
    # # print("success_ratio_25: {}".format(success_ratio_25))




