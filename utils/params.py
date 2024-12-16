import argparse
import os


def get_default_params(model_name):
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


def parse_args(args):
    # 用于创建一个命令行参数解析器, "Uni3D training and evaluation" 是程序的描述信息，会显示在帮助文档中。
    parser = argparse.ArgumentParser("Uni3D training and evaluation")

    # 还需要 nproc-per-node
    parser.add_argument(
        "--model", default="create_uni3d", type=str, help="Base model to create."
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="RN50",
        help="Name of the clip model to use.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="openai",
        help="Path to the pretrained model or model name (e.g., laion2b_s9b_b144k).",
    )
    

    parser.add_argument(
        "--pc-model",
        type=str,
        default="eva02_base_patch14_448",
        help="Name of pointcloud backbone to use.",
    )

   
    # Checkpoint path
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="checkpoints/model.pt",
        help="Path to the model checkpoint.",
    )

    parser.add_argument(
        "--seg_cat",
        type=str,
        default="airplane",
        help="which category do you want segmente.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="seg_res",
        help="which category do you want segmente.",
    )

    
    parser.add_argument(
        "--pretrained-pc",
        default='',
        type=str,
        help="Use a pretrained CLIP model vision weights with the specified tag or file path.",
    )

    
    parser.add_argument(
        "--lock-pointcloud",
        default=False,
        action="store_true",
        help="Lock full pointcloud's clip tower by disabling gradients.",
    )

    # Training
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log file on local master,otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU.")
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )

    # 学习率
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument(
        "--text-lr", type=float, default=None, help="Learning rate of text encoder."
    )
    parser.add_argument(
        "--visual-lr", type=float, default=None, help="Learning rate of visual encoder."
    )
    parser.add_argument(
        "--point-lr",
        type=float,
        default=None,
        help="Learning rate of pointcloud encoder.",
    )

    # Adam 超参数，可不了解
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")

    # Learning Rate 决定了模型参数的更新速度。
    # Weight Decay 则通过正则化的方式限制参数大小，辅助学习。
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--text-wd", type=float, default=None, help="Weight decay of text encoder."
    )
    parser.add_argument(
        "--visual-wd", type=float, default=None, help="Weight decay of visual encoder."
    )
    parser.add_argument(
        "--point-wd",
        type=float,
        default=None,
        help="Weight decay of pointcloud encoder.",
    )

    # Layer Decay 赋予每层不同的学习率:
    # 底层参数用较小学习率（更新慢，保持稳定）。
    # 顶层参数用较大学习率（更新快，快速适应新任务）。
    parser.add_argument(
        "--ld", type=float, default=1.0, help="Learning rate Layer decay."
    )
    parser.add_argument(
        "--text-ld",
        type=float,
        default=None,
        help="Learning rate Layer decay of text encoder.",
    )
    parser.add_argument(
        "--visual-ld",
        type=float,
        default=None,
        help="Learning rate Layer decay of visual encoder.",
    )
    parser.add_argument(
        "--point-ld",
        type=float,
        default=None,
        help="Learning rate Layer decay of pointcloud encoder.",
    )
    parser.add_argument(
        "--patch-dropout", type=float, default=0.0, help="flip patch dropout."
    )

    # 在训练的初始阶段，逐步增加学习率，避免一开始学习率过大引发不稳定或收敛困难。
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )

    # Batch Norm Sync 跨所有 GPU 或节点同步计算全局的均值和方差，确保每个设备使用的是全局一致的统计量。
    # 适合小 batch size 的分布式训练，因为同步计算提高了统计量的准确性。
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.",
    )

    # 决定是否在训练过程中启用学习率的衰减或其他调度策略。
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )

    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )

    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )

    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="amp",
        help="Floating point precision.",
    )

    # 允许用户自定义图像的均值（mean）值，用于图像预处理
    parser.add_argument(
        "--image-mean",
        type=float,
        nargs="+",
        default=None,
        metavar="MEAN",
        help="Override default image mean value of dataset",
    )
    parser.add_argument(
        "--image-std",
        type=float,
        nargs="+",
        default=None,
        metavar="STD",
        help="Override default image std deviation of of dataset",
    )

    # 有选择性地保存部分激活值，在需要反向传播时重新计算未保存的激活值，以换取显存节省
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action="store_true",
        help="Enable gradient checkpointing.",
    )

    # 是否以局部特征计算损失，而不是使用全局特征计算全局矩阵
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)",
    )

    # 是否启用带梯度的全分布式特征聚合
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather",
    )

    # 控制训练时的 patch dropout 值，可用于微调时设置较小或无 dropout。
    parser.add_argument(
        "--force-patch-dropout",
        default=None,
        type=float,
        help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",
    )

    # 设置分布式训练的初始化 URL，用于指定分布式环境的连接方式。
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )

    # 设置分布式训练的通信后端，默认使用 NCCL（NVIDIA GPU 的高效通信库）。
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )

    # 启用 WandB（Weights and Biases）日志记录功能。
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")

    # 设置 WandB 日志的运行 ID，用于恢复最新的检查点。
    parser.add_argument(
        "--wandb-runid",
        default=None,
        type=str,
        help="wandb runid to latest checkpoint (default: none)",
    )

    # 为 WandB 日志添加备注信息。
    parser.add_argument(
        "--wandb-notes", default="", type=str, help="Notes if logging with wandb"
    )

    # 指定 WandB 项目的名称，默认值为 'open-clip'。
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="open-clip",
        help="Name of the project if logging with wandb.",
    )

    # 如果启用此选项，记录更详细的调试信息。
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged.",
    )

    # 如果启用此选项，将整个代码库复制到日志目录，并从该目录中运行代码。
    # 在某些实验场景中，为了确保代码、配置、日志等可以完全追踪和复现，可能需要将运行的代码版本保存下来。
    # 直接运行代码时，如果代码库有更新或修改，后续很难复现实验结果。
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log diretory, and execute from there.",
    )

    # 启用 PyTorch >= 1.11 中 DDP（DistributedDataParallel）的静态图优化，可以提高性能和降低通信开销。
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action="store_true",
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )

    # 禁止根据 local rank 设置设备索引（适用于 CUDA_VISIBLE_DEVICES 限制为每个进程一个设备的情况）。
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )

    # 设置默认的随机种子，用于保证实验的可重复性。
    parser.add_argument("--seed", type=int, default=0, help="Default random seed.")

    # 设置梯度裁剪的最大范数（gradient clipping），防止梯度爆炸。
    parser.add_argument(
        "--grad-clip-norm", type=float, default=None, help="Gradient clip."
    )

    # 设置梯度累积的步数，适用于梯度累积的训练模式（目前仅支持 DeepSpeed）。
    parser.add_argument(
        "--grad-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps; only support deepspeed now.",
    )

    parser.add_argument("--start-epoch", default=0, type=int)

    # 优化器的更新频率，即梯度累积的步数，默认为 1。
    parser.add_argument(
        "--update-freq",
        default=1,
        type=int,
        help="optimizer update frequency (i.e. gradient accumulation steps)",
    )

    # 设置 Dropout 的丢弃概率，默认为 0。
    parser.add_argument("--drop-rate", default=0.0, type=float)

    # 设置 Drop Path 的丢弃概率，主要用于正则化深度学习模型，默认为 0。
    parser.add_argument("--drop-path-rate", default=0.0, type=float)

    # 设置模型评估的频率（每多少个 epoch 评估一次），默认为 1。
    parser.add_argument("--eval-freq", default=1, type=int)

    # 禁用混合精度训练（AMP），启用后需要更多的内存和计算资源。
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="disable mixed-precision training (requires more memory and compute)",
    )

    # 设置标签平滑的参数，默认为 0（无标签平滑），用于缓解模型过拟合。
    parser.add_argument(
        "--smoothing", type=float, default=0, help="Label smoothing (default: 0.)"
    )

    # 设置模型检查点的缓存目录，默认为 None。
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Default cache dir to cache model checkpoint.",
    )

    # 设置默认的优化器类型，默认为 'adamw'。
    parser.add_argument(
        "--optimizer", type=str, default="adamw", help="Default optimizer."
    )

    # 启用 DeepSpeed 框架，默认关闭。
    parser.add_argument("--enable-deepspeed", action="store_true", default=False)

    # 设置 DeepSpeed 的 ZeRO 优化阶段，默认为 1（基础阶段）。
    parser.add_argument("--zero-stage", type=int, default=1, help="stage of ZERO")

    # 启用图像和文本的嵌入（embedding）功能，默认关闭。
    parser.add_argument(
        "--use-embed",
        action="store_true",
        default=False,
        help="Use embeddings for iamge and text.",
    )

    # 是否使用大型 MiniPointNet 模型结构，默认关闭。
    parser.add_argument(
        "--is-large",
        action="store_true",
        default=False,
        help="whether to use large minipointnet",
    )

    # 设置保存嵌入的步数间隔，默认为 100。
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Step interval to store embeddings",
    )

    # 设置打印日志的频率（多少步打印一次），默认为 10。
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")

    # Data
    # 设置输出目录，用于存储结果，默认为 './outputs'。
    parser.add_argument(
        "--output-dir", default="./outputs", type=str, help="output dir"
    )

    # 设置预训练数据集的名称，默认为 'shapenet'。
    parser.add_argument("--pretrain_dataset_name", default="shapenet", type=str)

    # 设置预训练数据集的提示信息，默认为 'shapenet_64'。
    parser.add_argument("--pretrain_dataset_prompt", default="shapenet_64", type=str)

    # 设置验证数据集的名称，默认为 'modelnet40'。
    parser.add_argument(
        "--validate_dataset_name", default="modelnet40_openshape", type=str
    )

    # 设置验证数据集（Objaverse LVIS）的名称，默认为 'objaverse_lvis'。
    parser.add_argument(
        "--validate_dataset_name_lvis", default="objaverse_lvis_openshape", type=str
    )

    # 设置验证数据集（ScanObjNN）的名称，默认为 'scanobjnn_openshape'。
    parser.add_argument(
        "--validate_dataset_name_scanobjnn", default="scanobjnn_openshape", type=str
    )

    # 设置验证数据集的提示信息，默认为 'modelnet40_64'。
    parser.add_argument("--validate_dataset_prompt", default="modelnet40_64", type=str)

    # 是否使用 OpenShape 的 osaug 数据增强，默认关闭（在 OpenShape 数据集时自动启用）。
    parser.add_argument(
        "--openshape_setting",
        action="store_true",
        default=False,
        help="whether to use osaug, by default enabled with openshape.",
    )

    # 是否使用 LVIS 数据集，默认关闭。
    parser.add_argument(
        "--use_lvis",
        action="store_true",
        default=False,
        help="whether to use livs dataset.",
    )

    # point cloud
    # 设置点云中用于预训练和测试的点数量，默认为 8192。
    parser.add_argument(
        "--npoints",
        default=10000,
        type=int,
        help="number of points used for pre-train and test.",
    )

    # 是否使用高度信息（Height Information），默认为关闭（PointNeXt 模型默认启用）。
    parser.add_argument(
        "--use_height",
        action="store_true",
        default=False,
        help="whether to use height informatio, by default enabled with PointNeXt.",
    )

    # 设置点云特征的维度，默认为 768。
    parser.add_argument(
        "--pc_feat_dim", type=int, default=768, help="Pointcloud feature dimension."
    )

    # 设置点云 Transformer 中的分组大小，默认为 64。
    parser.add_argument(
        "--group-size", type=int, default=96, help="Pointcloud Transformer group size."
    )

    # 设置点云 Transformer 中的分组数量，默认为 512。
    parser.add_argument(
        "--num-group",
        type=int,
        default=1024,
        help="Pointcloud Transformer number of groups.",
    )

    # 设置点云 Transformer 编码器的维度，默认为 512。
    parser.add_argument(
        "--pc-encoder-dim",
        type=int,
        default=512,
        help="Pointcloud Transformer encoder dimension.",
    )

    # 设置教师模型的嵌入维度，默认为 1024。
    parser.add_argument(
        "--embed_dim", type=int, default=1024, help="teacher embedding dimension."
    )

    # evaluation
    # 仅评估 3D 模型，默认为关闭。
    parser.add_argument("--evaluate_3d", action="store_true", help="eval 3d only")

    args = parser.parse_args(args)

    # 如果指定了 --cache-dir 参数，将其值设置为环境变量 TRANSFORMERS_CACHE
    if args.cache_dir is not None:
        os.environ["TRANSFORMERS_CACHE"] = args.cache_dir  # huggingface model dir

    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        # 设置 lr beta1 beta2 eps
        if getattr(args, name) is None:
            setattr(args, name, val)

    if args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig

            os.environ["ENV_TYPE"] = "deepspeed"  # 设置环境变量 ENV_TYPE 为 deepspeed
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("")
            exit(0)
    else:
        os.environ["ENV_TYPE"] = "pytorch"
        ds_init = None
    return args, ds_init
