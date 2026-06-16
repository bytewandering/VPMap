# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")

"""
This code is mainly the deformation process of our DSConv
"""


class ESConv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, extend_scope, morph,
                 if_offset, device):
        """
        The Dynamic Snake Convolution
        :param in_ch: input channel
        :param out_ch: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param morph: the morphology of the convolution kernel is mainly divided into two types
                        along the x-axis (0) and the y-axis (1) (see the paper for details)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        :param device: set on gpu
        """
        super(ESConv, self).__init__()
        # use the <offset_conv> to learn the deformable offset
        self.offset_conv = nn.Conv2d(in_ch, 1, 3, padding=1)
        self.bn = nn.BatchNorm2d(1)
        self.kernel_size = kernel_size

        # two types of the DSConv (along x-axis and y-axis)
        self.dsc_conv_x = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = device

    def forward(self, f):
        offset = self.offset_conv(f)
        offset = self.bn(offset)
        # We need a range of deformation between -1 and 1 to mimic the snake's swing
        offset = torch.tanh(offset)
        input_shape = f.shape
        dsc = DSC(input_shape, self.kernel_size, self.extend_scope, self.morph,
                  self.device)
        deformed_feature = dsc.deform_conv(f, offset, self.if_offset)
        if self.morph == 0:
            x = self.dsc_conv_x(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)
            return x
        else:
            x = self.dsc_conv_y(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)
            return x


# Core code, for ease of understanding, we mark the dimensions of input and output next to the code
class DSC(object):

    def __init__(self, input_shape, kernel_size, extend_scope, morph, device):
        self.num_points = kernel_size
        self.width = input_shape[2]
        self.height = input_shape[3]
        self.morph = morph
        self.device = device
        self.extend_scope = extend_scope  # offset (-1 ~ 1) * extend_scope

        # define feature map shape
        """
        B: Batch size  C: Channel  W: Width  H: Height
        """
        self.num_batch = input_shape[0]
        self.num_channels = input_shape[1]

    """
    input: offset [B,2*K,W,H]  K: Kernel size (2*K: 2D image, deformation contains <x_offset> and <y_offset>)
    output_x: [B,1,W,K*H]   coordinate map
    output_y: [B,1,K*W,H]   coordinate map
    """


    def _coordinate_map_3D(self, offset, if_offset):
        """
        Input:
        feature-[B,C,W,H]
        offset_angle-[B,2*K+1,W,H]
        K-卷积核的中的采样点个数的一半
        Output:
            
        """ 
        """初始化采样点位置"""
        B,_,W,H = offset.shape
        K = self.num_points//2
        
        original_coor_x = torch.arange(0,W)
        original_coor_y = torch.arange(0,H)
        #[W,H]
        original_coor_x,original_coor_y = torch.meshgrid([original_coor_x,original_coor_y])
        original_coor_x = original_coor_x.unsqueeze(0).unsqueeze(0).repeat([B,2*K+1,1,1]).to(self.device)
        original_coor_y = original_coor_y.unsqueeze(0).unsqueeze(0).repeat([B,2*K+1,1,1]).to(self.device)
        
        """根据offset图计算每个采样点的偏移量"""
        #[B,1,W,H]
        offset_angle = offset*torch.pi/2
        
        #[B,2*K+1,W,H]
        offset_angle = offset_angle.repeat([1,2*K+1,1,1])
        
        #[B,2*K+1,W,H]
        if self.morph == 0:
            offset_x = torch.arange(-K,K+1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(self.device)*torch.cos(offset_angle)
            offset_y = torch.arange(-K,K+1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(self.device)*torch.sin(offset_angle)
        elif self.morph == 1:
            offset_angle = offset_angle-torch.pi/2
            offset_x = torch.arange(-K,K+1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(self.device)*torch.cos(offset_angle)
            offset_y = torch.arange(-K,K+1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(self.device)*torch.sin(offset_angle)
        
        
        """根据offset确定采样点的位置"""
        #[B,2*K+1,W,H]
        sampled_coor_x = original_coor_x+offset_x
        sampled_coor_y = original_coor_y+offset_y
        sampled_coor_x = torch.clamp(sampled_coor_x,0,W-1)
        sampled_coor_y = torch.clamp(sampled_coor_y,0,H-1)
        
        if self.morph == 0:
            sampled_coor_x = sampled_coor_x.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            sampled_coor_x = sampled_coor_x.permute(0, 3, 1, 4, 2)
            sampled_coor_x = sampled_coor_x.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            sampled_coor_y = sampled_coor_y.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            sampled_coor_y = sampled_coor_y.permute(0, 3, 1, 4, 2)
            sampled_coor_y = sampled_coor_y.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            return sampled_coor_x, sampled_coor_y
            
        else:
            sampled_coor_x = sampled_coor_x.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            sampled_coor_x = sampled_coor_x.permute(0, 3, 1, 4, 2)
            sampled_coor_x = sampled_coor_x.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            sampled_coor_y = sampled_coor_y.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            sampled_coor_y = sampled_coor_y.permute(0, 3, 1, 4, 2)
            sampled_coor_y = sampled_coor_y.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            return sampled_coor_x, sampled_coor_y

    """
    input: input feature map [N,C,D,W,H]；coordinate map [N,K*D,K*W,K*H] 
    output: [N,1,K*D,K*W,K*H]  deformed feature map
    """

    def _bilinear_interpolate_3D(self, input_feature, y, x):
        y = y.reshape([-1]).float()
        x = x.reshape([-1]).float()

        zero = torch.zeros([]).int()
        max_y = self.width - 1
        max_x = self.height - 1

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)

        input_feature_flat = input_feature.flatten()
        input_feature_flat = input_feature_flat.reshape(
            self.num_batch, self.num_channels, self.width, self.height)
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)
        input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)
        dimension = self.height * self.width

        base = torch.arange(self.num_batch) * dimension
        base = base.reshape([-1, 1]).float()

        repeat = torch.ones([self.num_points * self.width * self.height
                             ]).unsqueeze(0)
        repeat = repeat.float()

        base = torch.matmul(base, repeat)
        base = base.reshape([-1])

        base = base.to(self.device)

        base_y0 = base + y0 * self.height
        base_y1 = base + y1 * self.height

        # top rectangle of the neighbourhood volume
        index_a0 = base_y0 - base + x0
        index_c0 = base_y0 - base + x1

        # bottom rectangle of the neighbourhood volume
        index_a1 = base_y1 - base + x0
        index_c1 = base_y1 - base + x1

        # get 8 grid values
        value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(self.device)
        value_c0 = input_feature_flat[index_c0.type(torch.int64)].to(self.device)
        value_a1 = input_feature_flat[index_a1.type(torch.int64)].to(self.device)
        value_c1 = input_feature_flat[index_c1.type(torch.int64)].to(self.device)

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y + 1)
        y1 = torch.clamp(y1, zero, max_y + 1)
        x0 = torch.clamp(x0, zero, max_x + 1)
        x1 = torch.clamp(x1, zero, max_x + 1)

        x0_float = x0.float()
        x1_float = x1.float()
        y0_float = y0.float()
        y1_float = y1.float()

        vol_a0 = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(self.device)
        vol_c0 = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(self.device)
        vol_a1 = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(self.device)
        vol_c1 = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(self.device)

        outputs = (value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 +
                   value_c1 * vol_c1)

        if self.morph == 0:
            outputs = outputs.reshape([
                self.num_batch,
                self.num_points * self.width,
                1 * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        else:
            outputs = outputs.reshape([
                self.num_batch,
                1 * self.width,
                self.num_points * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        return outputs

    def deform_conv(self, input, offset, if_offset):
        y, x = self._coordinate_map_3D(offset, if_offset)
        deformed_feature = self._bilinear_interpolate_3D(input, y, x)
        return deformed_feature




def efficient_deform_feature(feature:torch.Tensor, offset, K, morph):
    """
    Input:
    feature-[B,C,W,H]
    offset_angle-[B,2*K+1,W,H]
    K-卷积核的中的采样点个数的一半
    Output:
        
    """ 
    B,C,W,H = feature.shape
    
    device = feature.device


    """初始化采样点位置"""
    original_coor_x = torch.arange(0,W)
    original_coor_y = torch.arange(0,H)
    #[W,H]
    original_coor_x,original_coor_y = torch.meshgrid([original_coor_x,original_coor_y])
    original_coor_x = original_coor_x.unsqueeze(0).unsqueeze(0).repeat([B,2*K+1,1,1]).to(device)
    original_coor_y = original_coor_y.unsqueeze(0).unsqueeze(0).repeat([B,2*K+1,1,1]).to(device)
    
    """根据offset图计算每个采样点的偏移量"""
    #[B,1,W,H]
    offset_angle = offset*torch.pi/2
    
    #[B,2*K+1,W,H]
    offset_angle = offset_angle.repeat([1,2*K+1,1,1])
    
    #[B,2*K+1,W,H]
    if morph == 0:
        offset_x = torch.arange(-K,K+1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)*torch.cos(offset_angle)
        offset_y = torch.arange(-K,K+1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)*torch.sin(offset_angle)
    elif morph == 1:
        offset_angle = offset_angle-torch.pi/2
        offset_x = torch.arange(-K,K+1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)*torch.cos(offset_angle)
        offset_y = torch.arange(-K,K+1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)*torch.sin(offset_angle)
    
    
    """根据offset确定采样点的位置"""
    #[B,2*K+1,W,H]
    sampled_coor_x = original_coor_x+offset_x
    sampled_coor_y = original_coor_y+offset_y
    sampled_coor_x = torch.clamp(sampled_coor_x,0,W-1)
    sampled_coor_y = torch.clamp(sampled_coor_y,0,H-1)
    
    #[B,2*K+1,W,H,2]
    sampled_coor = torch.stack([sampled_coor_x,sampled_coor_y],dim=-1)
    
    
    # sampled_coor[:,:,:,:,0] = torch.where(sampled_coor[:,:,:,:,0]>W-1,(W-1)*torch.ones_like(sampled_coor[:,:,:,:,0]),sampled_coor[:,:,:,:,0])
    # sampled_coor[:,:,:,:,0] = torch.where(sampled_coor[:,:,:,:,0]<0,torch.zeros_like(sampled_coor[:,:,:,:,0]),sampled_coor[:,:,:,:,0])
    # sampled_coor[:,:,:,:,1] = torch.where(sampled_coor[:,:,:,:,1]>H-1,(H-1)*torch.ones_like(sampled_coor[:,:,:,:,1]),sampled_coor[:,:,:,:,1])
    # sampled_coor[:,:,:,:,1] = torch.where(sampled_coor[:,:,:,:,1]<0,torch.zeros_like(sampled_coor[:,:,:,:,1]),sampled_coor[:,:,:,:,1])
 
    """使用别人的插值过程"""
    sampled_feature = []
    for i in range(0,2*K+1):
        sampled_feature.append(F.grid_sample(feature,sampled_coor[:,i,:,:,:]))
    #[B,C,W,2*K+1,H]
    sampled_feature = torch.stack(sampled_feature,dim=3)
    sampled_feature = sampled_feature.flatten(start_dim=2,end_dim=3)
    
    # """自己写的插值过程"""
    # #[B,W,H,2*K+1]
    # inter_point_x_floor = torch.floor(sampled_coor[:,:,:,:,0]).type(torch.int64)
    # inter_point_x_ceil = torch.ceil(sampled_coor[:,:,:,:,0]).type(torch.int64)
    # inter_point_y_floor = torch.floor(sampled_coor[:,:,:,:,1]).type(torch.int64)
    # inter_point_y_ceil = torch.ceil(sampled_coor[:,:,:,:,1]).type(torch.int64)
    
    
    # inter_point_x_floor = torch.clamp(inter_point_x_floor,0,W-1)
    # inter_point_x_ceil = torch.clamp(inter_point_x_ceil,0,W-1)
    # inter_point_y_floor = torch.clamp(inter_point_y_floor,0,H-1)
    # inter_point_y_ceil = torch.clamp(inter_point_y_ceil,0,H-1)
    
    
    # #[B,W,H,2*K+1,2]
    # inter_point1 = torch.stack([inter_point_x_floor,inter_point_y_floor],dim=-1)
    # inter_point2 = torch.stack([inter_point_x_ceil,inter_point_y_floor],dim=-1)
    # inter_point3 = torch.stack([inter_point_x_floor,inter_point_y_ceil],dim=-1)
    # inter_point4 = torch.stack([inter_point_x_ceil,inter_point_y_ceil],dim=-1)
    
    # #[B,W,H,2*K+1]
    # alpha1 = torch.sqrt(torch.sum((sampled_coor-inter_point1)**2,dim=-1))
    # alpha2 = torch.sqrt(torch.sum((sampled_coor-inter_point2)**2,dim=-1))
    # alpha3 = torch.sqrt(torch.sum((sampled_coor-inter_point3)**2,dim=-1))
    # alpha4 = torch.sqrt(torch.sum((sampled_coor-inter_point4)**2,dim=-1))
    
    # #[B,W,H,2*K+1,4]
    # alpha = torch.stack([alpha1,alpha2,alpha3,alpha4],dim=-1)
    # alpha = alpha/alpha.sum(dim=-1, keepdim=True)
    # #[B,W,H,2*K+1,1,4]
    # alpha = alpha.unsqueeze(-2).to(device)
    
    # batch_indices = torch.arange(0,B).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat([1,W,H,2*K+1]).type(torch.int64)
    
    # #[B,W,H,2*K+1,C]
    # feature1 = feature[batch_indices,
    #                 :,
    #                 inter_point1[:,:,:,:,0],
    #                 inter_point1[:,:,:,:,1]].to(device)
    # feature2 = feature[batch_indices,
    #                 :,
    #                 inter_point2[:,:,:,:,0],
    #                 inter_point2[:,:,:,:,1]].to(device)
    # feature3 = feature[batch_indices,
    #                 :,
    #                 inter_point3[:,:,:,:,0],
    #                 inter_point3[:,:,:,:,1]].to(device)
    # feature4 = feature[batch_indices,
    #                 :,
    #                 inter_point4[:,:,:,:,0],
    #                 inter_point4[:,:,:,:,1]].to(device)
    
    # #[B,W,H,2*K+1,C,4]
    # feature_fuse = torch.stack([feature1,feature2,feature3,feature4],dim=-1)
    
    # #[B,W,H,2*K+1,C]
    # # feature_fuse = torch.sum(alpha*feature_fuse,dim=-1)
    # feature_fuse = torch.mean(feature_fuse,dim=-1)
    
    #[B,C,(2*K+1)*W,H]
    # feature_fuse = feature_fuse.permute(0,4,3,1,2).flatten(start_dim=2,end_dim=3)
    
    return sampled_feature



# Code for testing the DSConv
# if __name__ == '__main__':
#     os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     A = np.random.rand(4, 5, 6, 7)
#     # A = np.ones(shape=(3, 2, 2, 3), dtype=np.float32)
#     # print(A)
#     A = A.astype(dtype=np.float32)
#     A = torch.from_numpy(A)
#     # print(A.shape)
#     conv0 = DSConv(
#         in_ch=5,
#         out_ch=10,
#         kernel_size=15,
#         extend_scope=1,
#         morph=0,
#         if_offset=True,
#         device=device)
#     if torch.cuda.is_available():
#         A = A.to(device)
#         conv0 = conv0.to(device)
#     out = conv0(A)
#     print(out.shape)
#     print(out)