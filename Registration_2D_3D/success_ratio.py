import SimpleITK as sitk
import numpy as np
import torch
from matplotlib import pyplot as plt
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

    results_dir_path = "F:/backup/data/IAS_train_info/IAS_2D_3D_Registration/norm_data/PPHReg/7/results.json"


    """读取预测的json文件"""
    with open(results_dir_path,"r") as file:
        results_json = json.load(file)

    mTRE_count = np.array(results_json["mTRE_count"])
    
    success_ratio_1 = np.mean(np.where(mTRE_count<1,1,0))
    success_ratio_2 = np.mean(np.where(mTRE_count<2,1,0))
    success_ratio_3 = np.mean(np.where(mTRE_count<3,1,0))
    success_ratio_4 = np.mean(np.where(mTRE_count<4,1,0))
    success_ratio_5 = np.mean(np.where(mTRE_count<5,1,0))
    success_ratio_6 = np.mean(np.where(mTRE_count<6,1,0))
    success_ratio_7 = np.mean(np.where(mTRE_count<7,1,0))
    success_ratio_8 = np.mean(np.where(mTRE_count<8,1,0))
    success_ratio_9 = np.mean(np.where(mTRE_count<9,1,0))
    success_ratio_10 = np.mean(np.where(mTRE_count<10,1,0))

    """保存成功率统计"""
    results_json["success_ratio_1"] = success_ratio_1
    results_json["success_ratio_2"] = success_ratio_2
    results_json["success_ratio_3"] = success_ratio_3
    results_json["success_ratio_4"] = success_ratio_4
    results_json["success_ratio_5"] = success_ratio_5
    results_json["success_ratio_6"] = success_ratio_6
    results_json["success_ratio_7"] = success_ratio_7
    results_json["success_ratio_8"] = success_ratio_8
    results_json["success_ratio_9"] = success_ratio_9
    results_json["success_ratio_10"] = success_ratio_10

    with open(results_dir_path,"w") as f:
        json.dump(results_json,f,indent=4)
        print("json文件生成成功")




 


 



