import torch
import torch.nn as nn


"""
paper: 2d U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
url: https://arxiv.org/abs/1606.06650
"""
class UNet_2d_encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 down_sample = True) -> None:
        super().__init__()

        self.down_sample_flag = down_sample
        #下采样
        if self.down_sample_flag:
            self.down_sample = nn.MaxPool2d(kernel_size=2,stride=2)

        #卷积
        self.conv1 = nn.Conv2d(in_channels,int(out_channels/2),kernel_size=3,stride=1,padding=1)
        self.batch_norm1 = nn.BatchNorm2d(int(out_channels/2))

        #卷积
        self.conv2 = nn.Conv2d(int(out_channels/2),out_channels,kernel_size=3,stride=1,padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self,x):

        if self.down_sample_flag:
            x = self.down_sample(x)

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)

        return x

class UNet_2d_decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels) -> None:
        super().__init__()

        self.up_sample = nn.ConvTranspose2d(in_channels,in_channels,kernel_size=2,stride=2)
        
        self.conv1 = nn.Conv2d(int(1.5*in_channels),out_channels,kernel_size=3,stride=1,padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
    
    def forward(self,x,x_skip):
        x = self.up_sample(x)
        x = torch.concat([x_skip,x], dim=1)
        
        #卷积
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        #卷积
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)

        return x


class UNet_2D(nn.Module):
    def __init__(self,
                 in_channels,
                 feature_embed,
                 out_channels) -> None:
        super().__init__()

        self.encoder1 = UNet_2d_encoder(in_channels, feature_embed, down_sample=False)
        self.encoder2 = UNet_2d_encoder(feature_embed,2*feature_embed)
        self.encoder3 = UNet_2d_encoder(2*feature_embed,4*feature_embed)
        self.encoder4 = UNet_2d_encoder(4*feature_embed,8*feature_embed)

        self.decoder3 = UNet_2d_decoder(8*feature_embed, 4*feature_embed)
        self.decoder2 = UNet_2d_decoder(4*feature_embed, 2*feature_embed)
        self.decoder1 = UNet_2d_decoder(2*feature_embed, feature_embed)

        self.seg_head = nn.Conv2d(feature_embed,out_channels,kernel_size=1)

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        out1 = self.decoder3(x4, x3)
        out2 = self.decoder2(out1,x2)
        out3 = self.decoder1(out2,x1)

        out = self.seg_head(out3)

        return out

