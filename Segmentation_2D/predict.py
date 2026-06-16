import numpy as np
import cv2
import os
import torch

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.inferers import SliceInferer

from models.UNet.unet_2d import UNet_2D
from models.TransUNet.vit_seg_modeling import VisionTransformer as TransUNet
from models.TransUNet.vit_seg_modeling import CONFIGS as TransUNet_CONFIGS
from models.Swin_Unet.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
# from models.UMamba.nets.UMambaBot_2d import get_umamba_bot_2d_from_plans
# from models.UMamba.nets.UMambaEnc_2d import get_umamba_enc_2d_from_plans
from models.DSCNet.DSCNet import DSCNet
# from models.MyDSCNet.DSCNet import DSCNet
# from models.VMNet.ESCNet import ESCNet
# from models.VMNet.ablation.ablation1 import VMNet_ablation1
# from models.VMNet.ablation.ablation4 import VMNet_ablation4
# from models.VMNet.ablation.ablation5 import VMNet_ablation5
# from models.VMNet.VMNet import VMNet
from models.CSNet.csnet import CSNet


from utils.utils import AverageMeter

import argparse
import time
import json
from collections import OrderedDict

parser = argparse.ArgumentParser(description="Point Cloud segmentation pipeline")
   
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")#输入图像的体素通道数
parser.add_argument("--num_class", default=2, type=int, help="number of classes")#输出的类别数

#batch_size只能设为1
parser.add_argument("--batch_size", default=1, type=int, help="batch size")


parser.add_argument("--data_dir", default="D:/personal_files/paper_list_ZHN/3_2d_3d_reg_TBME/data/IAS_2D_DSA", type=str, help="dataset directory")
parser.add_argument("--dataset_json", default="dataset.json", type=str, help="dataset json file")

parser.add_argument("--if_zero_norm", default=False, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--a_interval", default=[0,255], type=float, help="a_min in ScaleIntensityRanged")#体素值归一化
parser.add_argument("--b_interval", default=[0.0,1.0], type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--roi_x", default=512, type=float, help="a_min in ScaleIntensityRanged")#体素值归一化
parser.add_argument("--roi_y", default=512, type=float, help="b_min in ScaleIntensityRanged")

parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")



#使用不同模型做预测需要改的地方1/3：模型参数路径
parser.add_argument("--model_param_path", default="./checkpoints/TransUNet/best_model.pt")

#使用不同模型做预测需要改的地方2/3：预测数据的保存路径
parser.add_argument("--pred_directory", default="./predictions/TransUNet/test", type=str)


def get_data(data_dir,
             subset_data_path_dict,
             index,
             if_zero_norm,
             a_interval,
             b_interval
             ):
    #读取完整图像(D,W,H)
    image_name = subset_data_path_dict[index]["image"].split("/")[-3]+"/"+subset_data_path_dict[index]["image"].split("/")[-2]
    image_path = os.path.join(data_dir,subset_data_path_dict[index]["image"])
    image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    

    #设置窗位窗宽
    image = np.where(image<a_interval[0],a_interval[0]*np.ones_like(image),image)
    image = np.where(image>a_interval[1],a_interval[1]*np.ones_like(image),image)
    
    image = cv2.resize(image,(512,512),interpolation=cv2.INTER_NEAREST)

    #归一化图像体素
    if if_zero_norm:
        mean = np.mean(image)
        std = np.std(image)
        image = (image-mean)/std
    else:
        image = (image-np.ones_like(image)*a_interval[0]) / (a_interval[1]-a_interval[0]) * (b_interval[1]-b_interval[0])
    
    
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    return image_tensor, image_name


def main():
    args = parser.parse_args()
    
    """
    读取数据
    """
    with open(os.path.join(args.data_dir,args.dataset_json),"r") as file:
        data_path_dict = json.load(file)
        subset_data_path_dict = data_path_dict["test"]
        
    """
    设置cuda
    """
    device_name = "cuda"
    device = torch.device(device_name)
    print(device)


    """
    模型设置
    """
    #使用不同模型做预测需要改的地方3/3：初始化模型（不同的模型需要进行修改）
    
    """UNet"""
    # model = UNet_2D(
    #     in_channels=args.in_channels,
    #     feature_embed=16,
    #     out_channels=args.num_class
    # )

    """TransUNet"""
    TransUNet_CONFIG = TransUNet_CONFIGS["R50-ViT-B_16"]
    TransUNet_CONFIG.n_classes = args.num_class
    TransUNet_CONFIG.n_skip = 3
    TransUNet_CONFIG.patches.grid = (int(args.roi_x / 16), int(args.roi_y / 16)) #16是vit_patches_size
    model = TransUNet(TransUNet_CONFIG, img_size=args.roi_x, num_classes=args.num_class)

    """SwinUNet"""
    # model = SwinTransformerSys(
    #     img_size=args.roi_x,
    #     window_size=8,
    #     num_classes = args.num_class
    # )

    """UMambaBot"""
    # # 注意使用mamba的时候需要将model和data设成float
    # model = get_umamba_bot_2d_from_plans(num_input_channels=args.in_channels,
    #                                      num_classes=args.num_class,
    #                                      deep_supervision=False)
    
    """UMambaEnc"""
    # # 注意使用mamba的时候需要将model和data设成float
    # model = get_umamba_enc_2d_from_plans(num_input_channels=args.in_channels,
    #                                      num_classes=args.num_class,
    #                                      deep_supervision=False)

    """CSNet"""
    # model = CSNet(classes=args.num_class,channels=args.in_channels)

    """DSCNet"""
    # model = DSCNet(n_channels=args.in_channels,
    #                n_classes=args.num_class,
    #                kernel_size=9,
    #                extend_scope=1.0,
    #                if_offset=True,
    #                device="cuda",
    #                number=16,
    #                dim=1)

    """ESCNet"""
    # model = ESCNet(n_channels=args.in_channels,
    #                n_classes=args.num_class,
    #                kernel_size=9,
    #                extend_scope=1.0,
    #                if_offset=True,
    #                device="cuda",
    #                number=16,
    #                dim=1)

    """VMNet_ablation1"""
    # model = VMNet_ablation1(n_channels=args.in_channels,
    #                n_classes=args.num_class,
    #                kernel_size=9,
    #                extend_scope=1.0,
    #                if_offset=True,
    #                device="cuda",
    #                number=16,
    #                dim=1)
    
    """VMNet_ablation4"""
    # model = VMNet_ablation4(n_channels=args.in_channels,
    #                n_classes=args.num_class,
    #                kernel_size=9,
    #                extend_scope=1.0,
    #                if_offset=True,
    #                device="cuda",
    #                number=16,
    #                dim=1)

    """VMNet_ablation5"""
    # model = VMNet_ablation5(n_channels=args.in_channels,
    #                n_classes=args.num_class,
    #                kernel_size=9,
    #                extend_scope=1.0,
    #                if_offset=True,
    #                device="cuda",
    #                number=16,
    #                dim=1)

    """VMNet"""
    # model = VMNet(n_channels=args.in_channels,
    #                n_classes=args.num_class,
    #                kernel_size=9,
    #                extend_scope=1.0,
    #                if_offset=True,
    #                device="cuda",
    #                number=16,
    #                dim=1)
    
    #加载模型参数
    model_dict = torch.load(args.model_param_path,map_location="cpu")["state_dict"]
    model.load_state_dict(model_dict)
    model.to(device)
    model.float()
    model.eval()

    # #计算模型FlOPs
    # num_patches = (512-args.roi_x)/((1-args.infer_overlap)*args.roi_x) * (512-args.roi_y)/((1-args.infer_overlap)*args.roi_y) * (512-args.roi_z)/((1-args.infer_overlap)*args.roi_z)
    # tensor = torch.rand(args.sw_batch_size,1,args.roi_x,args.roi_y,args.roi_z)
    # flops = FlopCountAnalysis(model,tensor)
    # print("number of patches: ",num_patches/args.sw_batch_size)
    # print("GFLOPs:{:.2f} ".format(num_patches/args.sw_batch_size * flops.total()/10**9))

    
    """
    初始化存放推理时间的累积器
    """
    run_inference_time = AverageMeter()
    infer_time_dict_list = []

    

    with torch.no_grad():
        for i in range(len(subset_data_path_dict)):

            """
            获取模型的输入数据
            data - Tensor(B, C+C, D, H, W) 由两部分组成, 
                data[:, 0:C, 0:128, 0:128, 0:128]是完整的体数据
                data[:, C:, :, :, :]是待分割成子块的体数据
            (不同的数据处理需要修改)
            """
            print("Inference on case {}".format(i))

            
            #获取待分割图像、标签和全局图像
            data, image_name  = get_data(args.data_dir,
                                         subset_data_path_dict,
                                         i,
                                         args.if_zero_norm,
                                         args.a_interval,
                                         args.b_interval)

        
            #将数据存放在device上
            data= data.to(device)
            data = data.float()

            """
            模型预测
            """
            #在测试数据上进行预测
            start_time = time.time()
            pred = model(data)
            infer_time = time.time()-start_time
            run_inference_time.update(infer_time)
            print("inference time: {:.2f}s".format(infer_time))
            infer_time_dict = {
                image_name: infer_time
            }
            infer_time_dict_list.append(infer_time_dict)

     

            """
            保存模型输出的分割图像
            """
            #将pred由[1,C,D,W,H]变为[D,W,H]
            pred = pred.detach().cpu().squeeze(0)
            transform = AsDiscrete(argmax=True)
            pred = transform(pred)
            pred = pred.squeeze(0)
            
            pred_array = np.array(pred)

            if not os.path.exists(args.pred_directory+"/"+image_name):
                os.makedirs(args.pred_directory+"/"+image_name)
            cv2.imwrite(args.pred_directory+"/"+image_name+"/seg.png",pred_array)
            
            image = data.detach().cpu().squeeze(0).squeeze(0)*255
            plt.figure()
            plt.tight_layout()
            plt.axis("off")
            plt.imshow(image,cmap="gray")
            cmap = ListedColormap(["none","red"])
            plt.imshow(pred_array,
                       cmap=cmap, 
                       alpha=0.5, 
                       vmin=np.min(pred_array), 
                       vmax=np.max(pred_array))
            plt.savefig(args.pred_directory+"/"+image_name+"/show.png", dpi=512, bbox_inches='tight')
            
            print("Writing  done!")
            


    """打印输出最终信息"""
    print("Final Validation")
    
    #输出在数据集上的平均推理时间
    print("infer_time_avg: {:.2f}s".format(run_inference_time.avg))
    
    """将预测时间写入json文件"""
    infer_json = OrderedDict()
    infer_json["model_param_path"] = args.model_param_path
    infer_json["pred_directory"] = args.pred_directory
    infer_json["inference time"] = infer_time_dict_list
    infer_json["inference time avg"] = np.array(run_inference_time.vals).mean()
    infer_json["inference time std"] = np.array(run_inference_time.vals).std()
    with open(args.pred_directory+"/inference_time_{}.json".format(device_name),"w") as f:
        json.dump(infer_json,f,indent=4)
        print("json文件生成成功")


if __name__ == "__main__":
    main()