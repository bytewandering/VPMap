from typing import Any
import json
import os
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from .samplers import Sampler


"""
训练时生成的数据集为每个类别中选取num_patches个点, 以选取的点为中心点, 生成patches, 
__getitem__返回的数据的维度为[num_class*num_patches,C,D,W,H]

验证时生成的数据集为原始影像, __getitem__返回的数据的维度为[C,D,W,H]
"""


class IAS_2D_DSA(Dataset):
    
    
    def __init__(self,
                 data_dir:str,
                 dataset_json:str,
                 subset_name,
                 if_train=True,
                 if_zero_norm=False,
                 a_interval=[0,255],b_interval=[0,1],
                 image_name="image",label_name="label",
                 num_class=2,
                 num_slices=2) -> None:
        super().__init__()
        """
        Input:
        data_dir - 存放数据的文件夹
        dataset_json - 数据的json文件名
        subset_name - json文件中的子集名称,"training","validation"或"test"
        if_train - 是否是训练集，如果是训练集则对其进行切块处理
        if_zero_norm - 是否使用零均值归一化
        a_interval - 原始影像的截断像素区间
        b_interval - 不适用零均值归一化的情况下的像素映射区间
        image_name - 图像的名称
        label_name - 标签的名称
        num_class - 类别个数
        num_slices - 每个类别采样的patch数
        patch_size - patch的大小
        """
        
        self.data_dir = data_dir
        with open(os.path.join(data_dir,dataset_json),"r") as file:
            data_path_dict = json.load(file)
            self.subset_data_path_dict = data_path_dict[subset_name]

        self.if_train = if_train
        self.if_zero_norm = if_zero_norm

        self.a_interval = a_interval
        self.b_interval = b_interval

        self.image_name = image_name
        self.label_name = label_name

        
    def __len__(self):

        return len(self.subset_data_path_dict)
    
    def __getitem__(self, index: Any) -> Any:

        #读取完整图像(D,W,H)
        image_path = os.path.join(self.data_dir,self.subset_data_path_dict[index][self.image_name])
        image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

        #读取完整标签(D,W,H)
        label_path = os.path.join(self.data_dir,self.subset_data_path_dict[index][self.label_name])
        label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)


        #设置窗位窗宽
        image = np.where(image<self.a_interval[0],self.a_interval[0]*np.ones_like(image),image)
        image = np.where(image>self.a_interval[1],self.a_interval[1]*np.ones_like(image),image)

        if self.subset_data_path_dict[index][self.label_name].split("/")[0] == "cor":
            label = np.where(label>0,1,0)
        elif self.subset_data_path_dict[index][self.label_name].split("/")[0] == "sag":
            label = np.where(label==1,1,0)
        
        image = cv2.resize(image,(512,512),interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label,(512,512),interpolation=cv2.INTER_NEAREST)        

        #归一化图像体素
        if self.if_zero_norm:
            mean = np.mean(image)
            std = np.std(image)
            image = (image-mean)/std
        else:
            image = (image-np.ones_like(image)*self.a_interval[0]) / (self.a_interval[1]-self.a_interval[0]) * (self.b_interval[1]-self.b_interval[0])



            
        #(1,W,H)
        image = torch.from_numpy(image).unsqueeze(0)
        label = torch.from_numpy(label).unsqueeze(0)
        
        

        return image, label



            
def get_loader(args):
    train_dataset =  IAS_2D_DSA(args.data_dir,
                                            args.dataset_json,
                                            subset_name="train",
                                            if_zero_norm=args.if_zero_norm,
                                            a_interval=args.a_interval,
                                            b_interval=args.b_interval)
    
    train_sampler = Sampler(train_dataset) if args.distributed else None
    train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.world_size,
            sampler=train_sampler,
            pin_memory=True,
        )

    valid_dataset =  IAS_2D_DSA(args.data_dir,
                                            args.dataset_json,
                                            subset_name="test",
                                            if_train=False,
                                            if_zero_norm=args.if_zero_norm,
                                            a_interval=args.a_interval,
                                            b_interval=args.b_interval)
    val_sampler = Sampler(valid_dataset, shuffle=False) if args.distributed else None
    valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=1, 
            shuffle=False, 
            num_workers=args.world_size, 
            sampler=val_sampler, 
            pin_memory=True
        )

    return train_dataloader,valid_dataloader


if __name__ == "__main__":
    data_dir = "/root/data1/zhaohaining/IAS/IAS_2D_DSA"
    dataset_json = "dataset.json"
    dataset = IAS_2D_DSA(data_dir,dataset_json,subset_name="train",label_name="label")

    dataloader = DataLoader(dataset)

    for i, data in enumerate(dataloader):
        image, label = data