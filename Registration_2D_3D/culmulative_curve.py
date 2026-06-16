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



    pred_dir = "F:/backup/data/IAS_train_info/IAS_2D_3D_Registration/norm_data"
    out_dir = "F:/backup/data/IAS_train_info/IAS_2D_3D_Registration"
    method_list = os.listdir(pred_dir)
    k = 9


    culmulative_curves = []
    labels = []
    for method in method_list:

        method_dir = pred_dir+"/"+method+"/{}".format(k)
        with open(os.path.join(method_dir,"results.json"),"r") as file:
            results_dict = json.load(file)
        
        sr_list = []

        for j in range(1,11):
            sr_list.append(results_dict["success_ratio_{}".format(j)])

        culmulative_curves.append(sr_list)
        labels.append(method)

    culmulative_curves = np.array(culmulative_curves)
    plot_label = ["ICP","ICP+PWO","PPHReg"]
    
    plt.rcParams['font.family'] = 'serif'  # 使用 serif 字体系列
    plt.rcParams['font.serif'] = ['Times New Roman']  # 指定 serif 字体为 Times New Roman
    plt.rcParams['font.weight'] = 'bold'  # 设置字体粗细为加粗
    plt.rcParams['font.size'] = 20  # 设置字体粗细为加粗
    plt.figure()
    for i in range(len(culmulative_curves)):
        if labels[i] != "ICP+HMS":
            plt.plot(np.arange(1,len(culmulative_curves[i])+1), culmulative_curves[i]*100,label = labels[i])
            plt.scatter(np.arange(1,len(culmulative_curves[i])+1), culmulative_curves[i]*100)
    plt.xlabel("mTRE (mm)", fontname='Times New Roman', fontweight='bold')
    plt.ylabel("Ratio (%)", fontname='Times New Roman', fontweight='bold',)
    plt.legend(loc='lower right')
    plt.savefig(out_dir+"/norm_data_culmulative_curves/{}_show.png".format(k), dpi=512, bbox_inches='tight')
