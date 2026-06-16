import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np
from monai.networks.utils import one_hot

from utils.acc_func import dice,hd95,mIoU,clDice,accuracy,precision,sensitivity,roc_auc
from utils.utils import AverageMeter

import os
import argparse
import json
from collections import OrderedDict


parser = argparse.ArgumentParser(description="Point Cloud segmentation pipeline")
   
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--num_class", default=2, type=int, help="number of classes")#输出的类别数
parser.add_argument("--data_dir", default="D:/personal_files/paper_list_ZHN/3_2d_3d_reg_TBME/data/IAS_2D_DSA", type=str, help="dataset directory")
parser.add_argument("--dataset_json", default="dataset.json", type=str, help="dataset json file")

#不同模型分割结果评价需要改的地方1/1
parser.add_argument("--pred_directory", default="./predictions/UNet/test", type=str)


def get_data(data_dir,
             subset_data_path_dict,
             index,
             ):
    #读取完整图像(D,W,H)
    image_name = subset_data_path_dict[index]["image"].split("/")[-3]+"/"+subset_data_path_dict[index]["image"].split("/")[-2]
    label_path = os.path.join(data_dir,subset_data_path_dict[index]["label"])
    label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
    
    if subset_data_path_dict[index]["image"].split("/")[0] == "cor":
        label = np.where(label>0,1,0)
    elif subset_data_path_dict[index]["image"].split("/")[0] == "sag":
        label = np.where(label==1,1,0)
        
    label = cv2.resize(label,(512,512),interpolation=cv2.INTER_NEAREST)        

    
    label_tensor = torch.from_numpy(label).unsqueeze(0).unsqueeze(0)
    return label_tensor, image_name


def main():
    args = parser.parse_args()

    """
    读取数据
    """
    with open(os.path.join(args.data_dir,args.dataset_json),"r") as file:
        data_path_dict = json.load(file)
        subset_data_path_dict = data_path_dict["test"]


    """
    初始化每种精度的累积器
    """
    #列表run_acc_dice中存放着每一类的平均dice
    run_acc_dice = []
    for c in range(args.num_class):
        run_acc_dice_per_class = AverageMeter()
        run_acc_dice.append(run_acc_dice_per_class)

    #列表run_acc_mIoU中存放着每一类的平均mIoU
    run_acc_mIoU = []
    for c in range(args.num_class):
        run_acc_mIoU_per_class = AverageMeter()
        run_acc_mIoU.append(run_acc_mIoU_per_class)
        
    #列表run_acc_clDice中存放着每一类的平均clDice
    run_acc_clDice = []
    for c in range(args.num_class):
        run_acc_clDice_per_class = AverageMeter()
        run_acc_clDice.append(run_acc_clDice_per_class)

    #列表run_acc_hd95中存放着每一类的平均hd95
    run_acc_hd95 = []
    for c in range(args.num_class):
        run_acc_hd95_per_class = AverageMeter()
        run_acc_hd95.append(run_acc_hd95_per_class)
        
    #列表run_acc_acc中存放着每一类的平均acc
    run_acc_acc = []
    for c in range(args.num_class):
        run_acc_acc_per_class = AverageMeter()
        run_acc_acc.append(run_acc_acc_per_class)

    #列表run_acc_acc中存放着每一类的平均acc
    run_acc_prec = []
    for c in range(args.num_class):
        run_acc_prec_per_class = AverageMeter()
        run_acc_prec.append(run_acc_prec_per_class)

    #列表run_acc_acc中存放着每一类的平均acc
    run_acc_sens = []
    for c in range(args.num_class):
        run_acc_sens_per_class = AverageMeter()
        run_acc_sens.append(run_acc_sens_per_class)

    #列表run_acc_auc中存放着每一类的平均auc
    run_acc_auc = []
    for c in range(args.num_class):
        run_acc_auc_per_class = AverageMeter()
        run_acc_auc.append(run_acc_auc_per_class)

    

    for i in range(len(subset_data_path_dict)):


        """读取标签和预测图像"""
        target,image_name = get_data(args.data_dir,
                                    subset_data_path_dict,
                                    i)
        pred = cv2.imread(args.pred_directory+"/"+image_name+"/seg.png",cv2.IMREAD_GRAYSCALE)
        pred = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
        pred = one_hot(pred,args.num_class)#(B, C, D, W, H)
        
        print("Evaluation on {}:".format(image_name))

        """
        计算精度:Dice, HD95
        """
        #计算每一类的dice值
        acc_dice = dice(pred,target)
        for k in range(args.num_class):
            run_acc_dice[k].update(acc_dice[k].item(), n=args.batch_size)
        

        #计算每一类的mIoU
        acc_mIoU = mIoU(pred, target)
        for k in range(args.num_class):
            run_acc_mIoU[k].update(acc_mIoU[k].item(), n=args.batch_size)

        #计算每一类的mIoU
        acc_clDice = clDice(pred, target)
        for k in range(args.num_class):
            run_acc_clDice[k].update(acc_clDice[k].item(), n=args.batch_size)

        #计算每一类的hd95值
        acc_hd95 = hd95(pred,target)
        for k in range(args.num_class):
            run_acc_hd95[k].update(acc_hd95[k].item(), n=args.batch_size)

        #计算每一类的auc值
        acc_acc = accuracy(pred,target)
        for k in range(args.num_class):
            run_acc_acc[k].update(acc_acc[k].item(), n=args.batch_size)
            
        #计算每一类的hd95值
        acc_prec = precision(pred,target)
        for k in range(args.num_class):
            run_acc_prec[k].update(acc_prec[k].item(), n=args.batch_size)
            
        #计算每一类的hd95值
        acc_sens = sensitivity(pred,target)
        for k in range(args.num_class):
            run_acc_sens[k].update(acc_sens[k].item(), n=args.batch_size)

        #计算每一类的auc值
        acc_auc = roc_auc(pred,target)
        for k in range(args.num_class):
            run_acc_auc[k].update(acc_auc[k].item(), n=args.batch_size)


        """
        输出相关信息
        """
        for k in range(args.num_class):
            print("dice {}: {:.4f}".format(k, acc_dice[k]),
                  "mIoU {}: {:.4f}".format(k, acc_mIoU[k]),
                  "clDice {}: {:.4f}".format(k, acc_clDice[k]),
                  "hd95 {}: {:.4f}".format(k, acc_hd95[k]),
                  "acc {}: {:.4f}".format(k, acc_acc[k]),
                  "prec {}: {:.4f}".format(k, acc_prec[k]),
                  "sens {}: {:.4f}".format(k, acc_sens[k]),
                  "auc {}: {:.4f}".format(k, acc_auc[k]),)
                



    """打印输出最终信息"""
    print("Final Validation")
    

    #计算并打印输出测试集数据分割dice总体均值
    acc_dice_avg = AverageMeter()
    for i in range(args.num_class):
        acc_dice_avg.update(run_acc_dice[i].avg)

    #计算测试集数据分割mIoU总体均值
    acc_mIoU_avg = AverageMeter()
    for i in range(args.num_class):
        acc_mIoU_avg.update(run_acc_mIoU[i].avg)

    #计算测试集数据分割clDice总体均值
    acc_clDice_avg = AverageMeter()
    for i in range(args.num_class):
        acc_clDice_avg.update(run_acc_clDice[i].avg)

    #计算测试集数据分割hd95总体均值
    acc_hd95_avg = AverageMeter()
    for i in range(args.num_class):
        acc_hd95_avg.update(run_acc_hd95[i].avg)

    #计算测试集数据分割auc总体均值
    acc_acc_avg = AverageMeter()
    for i in range(args.num_class):
        acc_acc_avg.update(run_acc_acc[i].avg)        
    
    #计算测试集数据分割auc总体均值
    acc_prec_avg = AverageMeter()
    for i in range(args.num_class):
        acc_prec_avg.update(run_acc_prec[i].avg)
        
    #计算测试集数据分割auc总体均值
    acc_sens_avg = AverageMeter()
    for i in range(args.num_class):
        acc_sens_avg.update(run_acc_sens[i].avg)

    #计算测试集数据分割auc总体均值
    acc_auc_avg = AverageMeter()
    for i in range(args.num_class):
        acc_auc_avg.update(run_acc_auc[i].avg)


    #输出所有类别整体的平均分割精度
    print("dice_avg: {:.4f}".format(acc_dice_avg.avg),
          "mIoU_avg: {:.4f}".format(acc_mIoU_avg.avg),
          "clDice_avg: {:.4f}".format(acc_clDice_avg.avg),
          "hd95_avg: {:.4f}".format(acc_hd95_avg.avg),
          "acc_avg: {:.4f}".format(acc_acc_avg.avg),
          "prec_avg: {:.4f}".format(acc_prec_avg.avg),
          "sens_avg: {:.4f}".format(acc_sens_avg.avg),
          "auc_avg: {:.4f}".format(acc_auc_avg.avg),)
    
    #输出各类的平均分割精度
    for k in range(args.num_class):
        print("dice {}: {:.4f}".format(k, run_acc_dice[k].avg),
              "mIoU {}: {:.4f}".format(k, run_acc_mIoU[k].avg),
              "clDice {}: {:.4f}".format(k, run_acc_clDice[k].avg),
              "hd95 {}: {:.4f}".format(k, run_acc_hd95[k].avg),
              "acc {}: {:.4f}".format(k, run_acc_acc[k].avg),
              "prec {}: {:.4f}".format(k, run_acc_prec[k].avg),
              "sens {}: {:.4f}".format(k, run_acc_sens[k].avg),
              "auc {}: {:.4f}".format(k, run_acc_auc[k].avg))
        
    """保存精度信息"""
    
    acc_dict = OrderedDict()
    acc_dict["pred_dir"] = args.pred_directory
    acc_dict["label_dir"] = args.data_dir
    acc_dict["json_file"] = args.dataset_json
    for k in range(args.num_class):
        acc_dict["Dice {}".format(k)] = run_acc_dice[k].vals
        acc_dict["Dice {} avg".format(k)] = np.array(run_acc_dice[k].vals).mean()
        acc_dict["Dice {} std".format(k)] = np.array(run_acc_dice[k].vals).std()
        
        acc_dict["Dice_cor {} avg".format(k)] = np.array(run_acc_dice[k].vals)[0:12].mean()
        acc_dict["Dice_cor {} std".format(k)] = np.array(run_acc_dice[k].vals)[0:12].std()
        acc_dict["Dice_sag {} avg".format(k)] = np.array(run_acc_dice[k].vals)[12:].mean()
        acc_dict["Dice_sag {} std".format(k)] = np.array(run_acc_dice[k].vals)[12:].std()
        
        acc_dict["mIoU {}".format(k)] = run_acc_mIoU[k].vals
        acc_dict["mIoU {} avg".format(k)] = np.array(run_acc_mIoU[k].vals).mean()
        acc_dict["mIoU {} std".format(k)] = np.array(run_acc_mIoU[k].vals).std()

        acc_dict["mIoU_cor {} avg".format(k)] = np.array(run_acc_mIoU[k].vals)[0:12].mean()
        acc_dict["mIoU_cor {} std".format(k)] = np.array(run_acc_mIoU[k].vals)[0:12].std()   
        acc_dict["mIoU_sag {} avg".format(k)] = np.array(run_acc_mIoU[k].vals)[12:].mean()
        acc_dict["mIoU_sag {} std".format(k)] = np.array(run_acc_mIoU[k].vals)[12:].std()   
        

        acc_dict["clDice {}".format(k)] = run_acc_clDice[k].vals
        acc_dict["clDice {} avg".format(k)] = np.array(run_acc_clDice[k].vals).mean()
        acc_dict["clDice {} std".format(k)] = np.array(run_acc_clDice[k].vals).std()
        
        acc_dict["clDice_cor {} avg".format(k)] = np.array(run_acc_clDice[k].vals)[0:12].mean()
        acc_dict["clDice_cor {} std".format(k)] = np.array(run_acc_clDice[k].vals)[0:12].std()
        acc_dict["clDice_sag {} avg".format(k)] = np.array(run_acc_clDice[k].vals)[12:].mean()
        acc_dict["clDice_sag {} std".format(k)] = np.array(run_acc_clDice[k].vals)[12:].std()

        acc_dict["hd95 {}".format(k)] = run_acc_hd95[k].vals
        acc_dict["hd95 {} avg".format(k)] = np.array(run_acc_hd95[k].vals).mean()
        acc_dict["hd95 {} std".format(k)] = np.array(run_acc_hd95[k].vals).std()
        
        acc_dict["hd95_cor {} avg".format(k)] = np.array(run_acc_hd95[k].vals)[0:12].mean()
        acc_dict["hd95_cor {} std".format(k)] = np.array(run_acc_hd95[k].vals)[0:12].std()
        acc_dict["hd95_sag {} avg".format(k)] = np.array(run_acc_hd95[k].vals)[12:].mean()
        acc_dict["hd95_sag {} std".format(k)] = np.array(run_acc_hd95[k].vals)[12:].std()
        
        acc_dict["acc {}".format(k)] = run_acc_acc[k].vals
        acc_dict["acc {} avg".format(k)] = np.array(run_acc_acc[k].vals).mean()
        acc_dict["acc {} std".format(k)] = np.array(run_acc_acc[k].vals).std()
        
        acc_dict["acc_cor {} avg".format(k)] = np.array(run_acc_acc[k].vals)[0:12].mean()
        acc_dict["acc_cor {} std".format(k)] = np.array(run_acc_acc[k].vals)[0:12].std()
        acc_dict["acc_sag {} avg".format(k)] = np.array(run_acc_acc[k].vals)[12:].mean()
        acc_dict["acc_sag {} std".format(k)] = np.array(run_acc_acc[k].vals)[12:].std()
        
        acc_dict["prec {}".format(k)] = run_acc_prec[k].vals
        acc_dict["prec {} avg".format(k)] = np.array(run_acc_prec[k].vals).mean()
        acc_dict["prec {} std".format(k)] = np.array(run_acc_prec[k].vals).std()
        
        acc_dict["prec_cor {} avg".format(k)] = np.array(run_acc_prec[k].vals)[0:12].mean()
        acc_dict["prec_cor {} std".format(k)] = np.array(run_acc_prec[k].vals)[0:12].std()
        acc_dict["prec_sag {} avg".format(k)] = np.array(run_acc_prec[k].vals)[12:].mean()
        acc_dict["prec_sag {} std".format(k)] = np.array(run_acc_prec[k].vals)[12:].std()

        acc_dict["sens {}".format(k)] = run_acc_sens[k].vals
        acc_dict["sens {} avg".format(k)] = np.array(run_acc_sens[k].vals).mean()
        acc_dict["sens {} std".format(k)] = np.array(run_acc_sens[k].vals).std()
        
        acc_dict["sens_cor {} avg".format(k)] = np.array(run_acc_sens[k].vals)[0:12].mean()
        acc_dict["sens_cor {} std".format(k)] = np.array(run_acc_sens[k].vals)[0:12].std()
        acc_dict["sens_sag {} avg".format(k)] = np.array(run_acc_sens[k].vals)[12:].mean()
        acc_dict["sens_sag {} std".format(k)] = np.array(run_acc_sens[k].vals)[12:].std()

        acc_dict["auc {}".format(k)] = run_acc_auc[k].vals
        acc_dict["auc {} avg".format(k)] = np.array(run_acc_auc[k].vals).mean()
        acc_dict["auc {} std".format(k)] = np.array(run_acc_auc[k].vals).std()
        
        acc_dict["auc_cor {} avg".format(k)] = np.array(run_acc_auc[k].vals)[0:12].mean()
        acc_dict["auc_cor {} std".format(k)] = np.array(run_acc_auc[k].vals)[0:12].std()
        acc_dict["auc_sag {} avg".format(k)] = np.array(run_acc_auc[k].vals)[12:].mean()
        acc_dict["auc_sag {} std".format(k)] = np.array(run_acc_auc[k].vals)[12:].std()

    with open(args.pred_directory+"/eval_info1.json","w") as f:
        json.dump(acc_dict,f,indent=4)
        print("json文件生成成功")
        
if __name__ == "__main__":
    main()