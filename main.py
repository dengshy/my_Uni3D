from collections import OrderedDict
import math
import time
import wandb

import torch.cuda.amp as amp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import collections

from data.datasets import *

from utils import utils
from utils.utils import get_dateset
from utils.tokenizer import SimpleTokenizer
from utils.distributed import (
    create_deepspeed_config,
    world_info_from_env,
    is_master,
    init_distributed_device,
)
from utils.params import parse_args
from utils.logger import setup_logging
from utils.scheduler import warmup_cosine_lr
from utils.optim import (
    create_optimizer,
    get_all_parameter,
    get_loss_scale_for_deepspeed,
    get_grad_norm_,
)
from datetime import datetime
import open_clip
import models.uni3d as models

import os
import logging

best_acc1 = 0


def random_seed():
    return


def compute_embedding():
    return


def main(args):
    args, ds_init = parse_args(args)
    global best_acc1

    if torch.cuda.is_available():
        # 启用 TensorFloat32 提高计算性能
        torch.backends.cuda.matmul.allow_tf32 = True
        # 启用cuDNN 的自动优化
        torch.backends.cudnn.benchmark = True
        # 允许使用非确定性算法
        torch.backends.cudnn.deterministic = False
        # 启用cuDNN中的TF32支持
        torch.backends.cudnn.allow_tf32 = True

    # 打印时间，模型等参数
    if args.name is None:
        args.name = "-".join(
            [
                datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
                f"model_{args.model}",
                f"lr_{args.lr}",
                f"b_{args.batch_size}",
                f"j_{args.workers}",
                f"p_{args.precision}",
            ]
        )
    else:
        args.name = "-".join([args.name, datetime.now().strftime("%Y_%m_%d-%H")])

    # 为 DeepSpeed 创建并保存必要的配置文件
    if ds_init is not None:
        dsconfg_path = os.path.join(os.getcwd(), "dsconfig", args.name)
        os.makedirs(dsconfg_path, exist_ok=True)
        create_deepspeed_config(args)

    # 设置随机种子
    random_seed(args.seed, 0)

    # 获取分布式训练信息
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    args.log_path = None

    # 为实验日志系统设置路径，同时检查是否已经存在同名的实验，从而防止覆盖之前的结果
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.mkdirs(log_base_path, exist_ok=True)
        log_filename = f"out-{args.rank}" if args.log_local else "out.log"
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path):
            logging.error(
                "Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # 设置logger等级
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # 初始化分布式训练设备环境
    device = init_distributed_device(args)

    # 输出设备信息到日志
    logging.info("Distributed Device Environment Initialized")
    logging.info(f"Device Type:{device.type}")
    logging.info(f"Device Index:{device.index}")

    # 如果为GPU则输出更多CUDA信息
    if device.type == "cuda" and torch.cuda.is_available():
        logging.info(f"Device Name:{torch.cuda.get_device_name(device.index)}")
        logging.info(f"Total CUDA Devices:{torch.cuda.device_count()}")
        logging.info(f"Current CUDA Device:{torch.cuda.current_device()}")

    # 初始化 WandB（Weights & Biases）一个用于跟踪、分析和共享实验过程和结果的库
    if args.wandb and is_master(args):
        assert wandb is not None, "Please install wandb"
        logging.debug("Starting wandb.")
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            notes=args.wandb_notes,
            config=vars(args),  # vars(args) 会将 args 对象转换为字典格式
            settings=wandb.Settings(start_method="fork"),
        )

    # AMP mixed-precision
    if args.precision == "fp16":
        logging.warning(
            "It is recommended to use AMP mixed-precision instead of FP16. "
            "FP16 support needs further verification and tuning, especially for train."
        )
    elif args.distributed:
        logging.warning(
            f"Running in distributed mode with multiple processes. Device:{args.device}.",
            f"Process (global:{args.rank},local {args.local_rank}),total{args.world_size}.",
        )
    else:
        logging.info(f"Running with a single process. Device {args.device}.")

    # create model
    logging.info("=> creating model:{}".format(args.model))
    # 得到Uni3d的实例
    model = getattr(models, args.model)
    model.to(device)
    model_without_ddp = model

    # 创建clip模型
    logging.info("=> create clip teacher...")
    # 返回三个值 clip_model, image_transform, text_transform
    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name=args.clip_model, pretrained=args.pretrained
    )

    # 验证模型
    if args.evaluate_3d:
        logging.info("=> evaluating...")
        test_zeroshot_3d(args, model, clip_model)
        return


def train():
    return


def test_zeroshot_3d_core():
    return


def test_zeroshot_3d(args, model, clip_model):
    # 加载uni3d模型权重
    checkpoint = torch.load(args.ckpt_path,map_location="cpu")
    logging.info(
        "loaded checkpoint {}".format(args.ckpt_path)
    )
    
    # 提取模型参数sd（state_dict）
    # 如果是以分布式训练的，那么开头为module，需要去掉[module.]前缀
    sd=checkpoint["module"]
    if not args.distributed and next(iter(sd.items()))[0].startwith("module"):
        sd={k[len("module."):]:v for k,v in sd.items()}
    
    #pytorch 内置方法，加载模型权重
    model.load_state_dict(sd)
    
    tokenizer=SimpleTokenizer()

class AverageMeter:
    pass


class ProgressMeter:
    pass


def accuracy():
    return
