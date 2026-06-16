# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.distributed
from monai.transforms import AsDiscrete
import nibabel as nib


def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.vals = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.vals.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def distributed_all_gather(
    tensor_list, 
    valid_batch_size=None, 
    out_numpy=False, 
    world_size=None, 
    no_barrier=False, 
    is_valid=None
):
    #对tensor_list中的每个tensor，从所有的GPU中收集他们，形成n_GPU个n_GPU的List[Tensor]（对应程序中的gather_list），并将其分发给各个GPU

    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out


def save_image(images:torch.Tensor,
               meta_dict, 
               save_path):
    """
    preds - torch.Tensor(B,C,D,H,W);
    original_affine - 原始图像的仿射矩阵;
    save_path - 保存路径;
    Attention - 目前batch size仅能设为1
    """

    #将概率图(C,D,H,W)改为分割图(1,D,H,W)的变换
    transform = AsDiscrete(argmax=True)

    #遍历批数据
    for image in images:

        #将概率图(C,D,H,W)改为分割图(D,H,W)
        image_discrete = transform(image)
        image_discrete = torch.squeeze(image_discrete,axis=0)
        image_discrete = image_discrete.cpu().numpy()

        #将数组的保存方向由RAS调整为原始图像的数组保存方向
        x,y,z = nib.aff2axcodes(meta_dict["original_affine"][0].numpy())
        if x == "L":
            image_discrete = np.flip(image_discrete, axis=0)

            if y == "A":
                if z == "S":
                    pass
                elif z == "I":
                    image_discrete = np.flip(image_discrete, axis=2)

            elif y == "P":
                image_discrete = np.flip(image_discrete,axis=1)
                if z == "S":
                    pass
                elif z == "I":
                    image_discrete = np.flip(image_discrete, axis=2)

            elif y == "S":
                image_discrete = np.transpose(image_discrete,(0,2,1))
                if z == "A":
                    pass
                elif z == "P":
                    image_discrete = np.flip(image_discrete,axis=2)
            elif y == "I":
                image_discrete = np.transpose(image_discrete,(0,2,1))
                image_discrete = np.flip(image_discrete,axis=1)
                if z == "A":
                    pass
                elif z == "P":
                    image_discrete = np.flip(image_discrete,axis=2)

        elif x == "R":
            if y == "A":
                if z == "S":
                    pass
                elif z == "I":
                    image_discrete = np.flip(image_discrete, axis=2)

            elif y == "P":
                image_discrete = np.flip(image_discrete,axis=1)
                if z == "S":
                    pass
                elif z == "I":
                    image_discrete = np.flip(image_discrete, axis=2)

            elif y == "S":
                image_discrete = np.transpose(image_discrete,(0,2,1))
                if z == "A":
                    pass
                elif z == "P":
                    image_discrete = np.flip(image_discrete,axis=2)
            elif y == "I":
                image_discrete = np.transpose(image_discrete,(0,2,1))
                image_discrete = np.flip(image_discrete,axis=1)
                if z == "A":
                    pass
                elif z == "P":
                    image_discrete = np.flip(image_discrete,axis=2)



        image_nib = nib.Nifti1Image(image_discrete.astype(np.uint8), meta_dict["original_affine"][0].numpy())

        #设置数据的元信息与原始数据保持一致        
        image_nib.header.set_xyzt_units(xyz=meta_dict["xyzt_units"].item())
        # image_nib.header.set_pixdim(meta_dict["pixdim"][0].numpy())
        image_nib.header.set_qform(image_nib.affine,code=meta_dict["qform_code"].item())
        image_nib.header.set_sform(image_nib.affine,code=meta_dict["sform_code"].item())


        #print(image_nib.header)
        nib.save(
            image_nib,
            save_path 
            )
