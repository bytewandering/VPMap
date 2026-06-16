import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import random
from functools import partial
from utils.learning_scheduler import LinearWarmupCosineAnnealingLR

from torch.cuda.amp import GradScaler, autocast
import time
import os
from torch.utils.tensorboard import SummaryWriter

from utils.utils import AverageMeter,distributed_all_gather

#数据导入
from dataloader.dataloader_2d import get_loader

#训练不同模型需要修改的地方1/4：模型声明
from models.UNet.unet_2d import UNet_2D
from models.TransUNet.vit_seg_modeling import VisionTransformer as TransUNet
from models.TransUNet.vit_seg_modeling import CONFIGS as TransUNet_CONFIGS
from models.Swin_Unet.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
# from models.UMamba.nets.UMambaBot_2d import get_umamba_bot_2d_from_plans
# from models.UMamba.nets.UMambaEnc_2d import get_umamba_enc_2d_from_plans
from models.DSCNet.DSCNet import DSCNet
# from models.MyDSCNet.DSCNet import DSCNet
from models.CSNet.csnet import CSNet
# from models.VMNet.ESCNet import ESCNet
# from models.VMNet.ablation.ablation1 import VMNet_ablation1
# from models.VMNet.ablation.ablation4 import VMNet_ablation4
# from models.VMNet.ablation.ablation5 import VMNet_ablation5
# from models.VMNet.VMNet import VMNet


#损失函数
from utils.loss_func import dice_ce_loss,dice_focal_loss,mean_dice_loss



#分割指标
from utils.acc_func import dice

"""命令行参数解析"""
parser = argparse.ArgumentParser(description="Point Cloud segmentation pipeline")

"""分布式环境参数"""
parser.add_argument("--distributed",action = "store_true", default=True, help="use distributed training")   
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--ngpus_per_node", default=8, type=int, help="number of GPUs of each node for distributed training, no need to set")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training, no need to set")
parser.add_argument("--dist_url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist_backend", default="nccl", type=str, help="distributed backend")

"""数据"""
parser.add_argument("--data_dir", default="D:/personal_files/paper_list_ZHN/3_2d_3d_reg_TBME/data/IAS_2D_DSA", type=str, help="dataset directory")
parser.add_argument("--dataset_json", default="dataset.json", type=str, help="dataset json file")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--use_normal_dataset", action="store_true", default=True, help="use monai Dataset class")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")#输入图像的体素通道数
parser.add_argument("--num_class", default=2, type=int, help="number of classes")#输出的类别数

#数据归一化方式
parser.add_argument("--if_zero_norm", default=False, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--a_interval", default=[0,255], type=float, help="a_min in ScaleIntensityRanged")#体素值归一化
parser.add_argument("--b_interval", default=[0.0,1.0], type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--roi_x", default=512, type=float, help="a_min in ScaleIntensityRanged")#体素值归一化
parser.add_argument("--roi_y", default=512, type=float, help="b_min in ScaleIntensityRanged")



"""模型"""
parser.add_argument("--pretrained_model", default=None, type=str, help="load pretrained model parameters(path)")
parser.add_argument("--checkpoint", default=None, type=str, help="start training from saved checkpoint(path)")

#训练不同模型需要修改的地方2/4：checkpoints存放路径
parser.add_argument("--checkpoint_dir", default="./checkpoints/TransUNet", type=str, help="directory saving checkpoint")

parser.add_argument("--norm_name", default="batch", type=str, help="normalization method name")
parser.add_argument("--pretrained_encoder", default=None, type=str, help="load pretrained encoder parameters(path)")
#parser.add_argument("--pretrained_encoder", default="./pretrained_models/model_swinvit.pt", type=str, help="load pretrained encoder parameters(path)")

"""sliding window infer"""
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")

"""优化器"""
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")

"""训练信息"""
parser.add_argument("--seed", default=42, type=int, help="set seed to fix random")
parser.add_argument("--start_epoch", default=0, type=int, help="epoch starting training")
parser.add_argument("--max_epochs", default=5000, type=int, help="max number of training epochs")
parser.add_argument("--val_every", default=10, type=int, help="when to validate")
parser.add_argument("--best_acc", default=0, type=float, help="best accuracy")
parser.add_argument("--print_rank", default=0, type=int, help="num of works used to print")

#训练不同模型需要修改的地方3/4：logdir路径
parser.add_argument("--log_dir", default="./logdir/TransUNet", type=str, help="directory to save log")

parser.add_argument("--amp", default=True, type=bool, help="whether to use auto mixed precision for training")




"""只计算一个损失的训练epoch"""
def train_epoch(epoch:int,
                model:nn.Module,
                loader:torch.utils.data.DataLoader,
                optimizer,
                scaler,
                loss_func,
                args):
    model.train()
    run_loss = AverageMeter()
    start_time = time.time()

    for i, batch_data in enumerate(loader):

        """读取数据"""
        data, target = batch_data
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        data, target = data.float(), target.float()
        # data, target = data.double(), target.double()
        

        """梯度清零"""
        for param in model.parameters():
            param.grad = None

        """前向计算, 反向传播, 更新参数（是否使用自动混合精度）"""
        if args.amp:
            with autocast(enabled=True):
                pred = model(data)
                loss = loss_func(pred, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(data)
            loss = loss_func(pred, target)
            
            loss.backward()
            optimizer.step()
        
        if args.distributed:
            """累计一轮中所有批次数据的损失"""
            loss_list = distributed_all_gather([loss], out_numpy=True)
            #loss_list-[[loss_GPU0, loss_GPU1, loss_GPU2, loss_GPU3]]
            run_loss.update(
                np.mean(np.stack(loss_list[0], axis=0), axis=0), 
                n=args.batch_size * args.world_size
            )

            """输出第epoch轮, 第i批次训练数据的训练情况"""
            if args.rank == args.print_rank:
                print(
                    "Epoch {}/{} {}/{}".format(epoch, args.max_epochs-1, i, len(loader)-1),
                    "loss: {:.4f}".format(np.mean(np.stack(loss_list[0], axis=0), axis=0)),
                    "time {:.2f}s".format(time.time() - start_time)
                )
        else:
            """累计一轮中所有批次数据的损失"""
            run_loss.update(
                loss.item(), 
                n=args.batch_size
            )

            """输出第epoch轮, 第i批次训练数据的训练情况"""
            if args.rank == args.print_rank:
                print(
                    "Epoch {}/{} {}/{}".format(epoch, args.max_epochs-1, i, len(loader)-1),
                    "loss: {:.4f}".format(loss.item()),
                    "time {:.2f}s".format(time.time() - start_time)
                )
        
    
    for param in model.parameters():
        param.grad = None
    
    return run_loss.avg

def val_epoch(epoch:int,
              model:nn.Module,
              loader:torch.utils.data.DataLoader,
              acc_func,
              model_infer,
              args):

    """
    计算第epoch轮的模型在验证集上每一类的精度
    Params
    ------
    Return
    ------
    acc_per_class: list[scalar]
    """

    model.eval()
    
    #run_acc中存放每一类的分割精度
    run_acc = []
    for c in range(args.num_class):
        run_acc_per_class = AverageMeter()
        run_acc.append(run_acc_per_class)

    
    start_time = time.time()

    with torch.no_grad():
        for i, batch_data in enumerate(loader):

            """读取数据"""
            data, target = batch_data
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            data, target = data.float(), target.float()
            # data, target = data.double(), target.double()

            """前向"""
            with autocast(enabled=args.amp):
                pred = model(data)
                acc = acc_func(pred, target)#acc-List[Tensor]，存储各个类别的在当前batch上的分割精度
            

            if args.distributed:

                """分布式累计分割精度"""
                #acc_list-
                # [[acc0_GPU0, acc0_GPU1, acc0_GPU2, acc0_GPU3],
                # [acc1_GPU0, acc1_GPU1, acc1_GPU2, acc1_GPU3],
                # [acc2_GPU0, acc2_GPU1, acc2_GPU2, acc2_GPU3]]
                acc_list = distributed_all_gather(acc, out_numpy=True)
                
                #将每一类在所有GPU上的平均分割精度累计到各自的run_acc[k]中
                for k in range(args.num_class):
                    run_acc[k].update(np.mean(np.stack(acc_list[k], axis=0), axis=0), n=args.batch_size * args.world_size)
                
                """输出第epoch轮, 第i批次测试数据上, 各类的分割精度"""
                if args.rank == args.print_rank:
                    print("Val {}/{} {}/{}".format(epoch, args.max_epochs-1, i, len(loader)-1))
                    for k in range(args.num_class):
                        print("acc {}: {:.4f}".format(k, np.mean(np.stack(acc_list[k], axis=0), axis=0)))
                    print("time {:.2f}s".format(time.time() - start_time))
            else:
                """累计分割精度"""
                #将每一类在所有GPU上的平均分割精度累计到各自的run_acc[k]中
                for k in range(args.num_class):
                    run_acc[k].update(acc[k].item(), n=args.batch_size)

                """输出第epoch轮, 第i批次测试数据上, 各类的分割精度"""
                if args.rank == args.print_rank:
                    print("Val {}/{} {}/{}".format(epoch, args.max_epochs-1, i, len(loader)-1))
                    for k in range(args.num_class):
                        print("acc {}: {:.4f}".format(k, acc[k].item()))
                    print("time {:.2f}s".format(time.time() - start_time))

    
    """返回第epoch轮, 测试数据上, 各类的分割精度"""
    epoch_acc = []
    for k in range(args.num_class):
        epoch_acc.append(run_acc[k].avg)
    return epoch_acc


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.checkpoint_dir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)    



def run_training(model:nn.Module, 
                 train_loader:torch.utils.data.DataLoader,
                 val_loader:torch.utils.data.DataLoader,
                 loss_func,
                 acc_func,
                 optimizer,
                 lr_scheduler,
                 model_infer,
                 args):
    """1、创建tensorboard输出器"""
    if args.log_dir is not None and args.rank == args.print_rank:
        writer = SummaryWriter(log_dir=args.log_dir)
        print("Writing Tensorboard logs to ", args.log_dir)
    else:
        writer = None 

    """2、自动混合精度"""
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    val_best_acc = args.best_acc
    for epoch in range(args.start_epoch, args.max_epochs):
        """3、训练"""
        #分布式训练设置
        if args.distributed:
            #每轮需要设置一下才会对数据进行打乱
            train_loader.sampler.set_epoch(epoch)
            #阻塞进程，等待所有进程到达此处再继续
            dist.barrier()
        
        #输出每个进程的每个epoch开始的时间
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()

        #训练
        train_loss = train_epoch(epoch,model,train_loader,optimizer,scaler,loss_func,args)

        #主进程输出训练信息
        if args.rank == args.print_rank:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
            if writer is not None:
                writer.add_scalar("train_loss", train_loss, epoch)

        """4、验证"""
        if (epoch+1)%args.val_every == 0:
            #分布式训练设置
            if args.distributed:
                dist.barrier()
            
            #验证开始时间
            epoch_time = time.time()

            #验证
            epoch_acc = val_epoch(epoch,model,val_loader,acc_func,model_infer,args)
            val_avg_acc = np.mean(np.array(epoch_acc))

            if args.rank == args.print_rank:
                print("Final training  {}/{}".format(epoch, args.max_epochs - 1))
                print("acc_avg: {:.4f}".format(val_avg_acc))
                for k in range(args.num_class):
                    print("acc_{}: {:.4f}".format(k, epoch_acc[k]))

                print("time {:.2f}s".format(time.time() - epoch_time))


                if writer is not None:
                    writer.add_scalar("acc_avg", val_avg_acc, epoch)
                    for k in range(args.num_class):
                        writer.add_scalar("acc_{}".format(k), epoch_acc[k], epoch)
                
                #保存最优模型
                if val_avg_acc > val_best_acc:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_best_acc, val_avg_acc))
                    val_best_acc = val_avg_acc
                    save_checkpoint(model,
                                    epoch,
                                    args,
                                    filename="best_model.pt",
                                    best_acc=val_best_acc,
                                    optimizer=optimizer,
                                    scheduler=lr_scheduler)
                                
                #保存最后一轮的模型模型
                if not np.isnan(train_loss):
                    save_checkpoint(model,
                                    epoch,
                                    args,
                                    best_acc=val_best_acc,
                                    optimizer=optimizer,
                                    scheduler=lr_scheduler)

            
        """5、更新优化器参数"""
        if lr_scheduler is not None:
            lr_scheduler.step()

    """6、输出最后的分割精度"""
    print("Training Finished !, Best Accuracy: ", val_best_acc)
    
    return val_best_acc


def main_worker(gpu, args):
    

    """
    初始化分布式环境
    Attention
    ---------
    DDP分布式训练需要设置四个部分:
    初始化分布式环境(init_process_group)、
    数据集划分到各个卡上(DistributedSampler)、
    设置各个卡上的模型参数的通信(SyncBatchNorm、DistributedDataParallel)
    启动(spawn)
    """
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)

    if args.distributed:
        #设置子进程构建模式
        torch.multiprocessing.set_start_method("fork", force=True)
        #计算当前GPU的序号(rank)，这里将rank由当前node的序号更新为了当前GPU的序号
        args.rank = args.rank * args.ngpus_per_node + args.gpu
        #backend是后端通讯方式， init_method是完成进程同步的初始化方式，world_size是所有GPU的个数(由node个数更新为所有GPU的个数)，rank是当前GPU的序号
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    
    """固定随机性"""
    #固定随机种子点
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    #固定卷积方式
    #torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    """
    设置数据集
    Attention
    ---------
    data:图像数据
        Tensor (B, C, D, H, W)
    target:图像标签 
        Tensor (B, 1, D, H, W)
    """
    train_dataloader, val_dataloader = get_loader(args)


    """
    设置模型
    Attention
    ---------
    1.forward(self,x)
    Param
    -----
    x:点云数据
        Tensor (B, C, D, H, W)
    Return
    ------
    pred:分割结果(未经过softmax层)
        Tensor (B, n_class, D, H, W)
    """
    #声明模型
    #训练不同模型需要修改的地方4/4：模型声明
    
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

    """UMamba_Bot"""
    # #注意使用mamba的时候需要将model和data设成float
    # model = get_umamba_bot_2d_from_plans(num_input_channels=args.in_channels,
    #                                      num_classes=args.num_class,
    #                                      deep_supervision=False)
    
    """UMamba_Enc"""
    # #注意使用mamba的时候需要将model和data设成float
    # model = get_umamba_enc_2d_from_plans(num_input_channels=args.in_channels,
    #                                      num_classes=args.num_class,
    #                                      deep_supervision=False)

    """DSCNet"""
    # model = DSCNet(n_channels=args.in_channels,
    #                n_classes=args.num_class,
    #                kernel_size=9,
    #                extend_scope=1.0,
    #                if_offset=True,
    #                device="cuda",
    #                number=16,
    #                dim=1)

    """CSNet"""
    # model = CSNet(classes=args.num_class,channels=args.in_channels)

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

    #加载预训练模型
    if args.pretrained_model is not None:
        state_dict = torch.load(args.pretrained_model)["state_dict"]
        model.load_state_dict(state_dict)


    #加载训练节点
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")

        #从checkpoint处加载模型参数
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=True)

        #从checkpoint处加载训练信息
        if "epoch" in checkpoint:
            args.start_epoch = checkpoint["epoch"] + 1
        if "best_acc" in checkpoint:
            args.best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, args.start_epoch, args.best_acc))

    #设置预测方法
    model_infer = None
    
    #输出模型参数量
    if args.rank == 0:
        num_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total parameters count: ", num_model_params)

    #分布式训练环境下实现真正的BN
    if args.distributed:
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    #将模型迁移到GPU上
    model.cuda(args.gpu)
    model.float()
    # model.double()

    #分布式训练，创建DDP模型
    if args.distributed:
        #初始化 Distributed Data Parallel模型
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.gpu], 
                                                          output_device=args.gpu,
                                                          find_unused_parameters=True)

    """
    设置损失函数和评价指标
    """
    #损失函数
    #pred-Tensor (B, n_class, D, H, W), target-Tensor (B, 1, D, H, W) -> loss-Scalar;
    loss_func = dice_ce_loss


    #评价指标
    #pred-Tensor (B, n_class, D, H, W), target-Tensor (B, 1, D, H, W) -> acc-Tensor (n_class)
    acc_func = dice

    """设置优化器"""
    #选择优化器
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    #选择学习率调整方法
    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=args.start_epoch)
    else:
        scheduler = None

    """训练"""
    run_training(
        model,
        train_dataloader,
        val_dataloader,
        loss_func,
        acc_func,
        optimizer,
        scheduler,
        model_infer,
        args)

if __name__ == "__main__":
    args = parser.parse_args()

    if torch.cuda.device_count() == 1:
        args.distributed = False

    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus: ", args.ngpus_per_node)

        #world size由节点总数更新为GPU总数
        args.world_size = args.ngpus_per_node * args.world_size

        #开启nprocs个进程（一个GPU上运行一个进程）来运行main_worker(i,*args)函数；
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)