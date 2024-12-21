import torch.utils
import torch.cuda.amp as amp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from data.datasets import *

# utils 与系统包重名，添加 __init__.py 文件
from utils import utils
from utils.utils import get_dataset
from utils.tokenizer import SimpleTokenizer
from utils.distributed import (
    create_deepspeed_config,
    world_info_from_env,
    is_master,
    init_distributed_device,
)
from utils.params import parse_args
from utils.logger import setup_logging
from datetime import datetime
import open_clip
import models.uni3d as models

import os
import logging
from utils.draw import save_point_cloud_html
from kmeans_pytorch import kmeans
import debugpy

try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass

# 全局变量
class_counters = {}


# 定义函数来更新计数器
def update_class_counter(class_name):
    global class_counters  # 使用 global 关键字
    if class_name not in class_counters:
        class_counters[class_name] = 1
    else:
        class_counters[class_name] += 1
    print(f"Class '{class_name}' Counter: {class_counters[class_name]}")


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


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
    random_seed(args.seed)

    # 获取分布式训练信息
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    args.log_path = None

    # 为实验日志系统设置路径，同时检查是否已经存在同名的实验，从而防止覆盖之前的结果
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
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

    # create model
    logging.info("=> creating model:{}".format(args.model))
    # 得到Uni3d的实例
    model = getattr(models, args.model)(args=args)
    model.to(device)

    # clip_model=None
    # 创建clip模型
    logging.info("=> create clip teacher...")
    # 返回三个值 clip_model, image_transform, text_transform
    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name=args.clip_model, pretrained=args.pretrained
    )
    clip_model.to(device)

    mn40_result = test_zeroshot_3d(args, model, clip_model)
    logging.info(mn40_result)
    return


def test_zeroshot_3d_core(
    test_loder,
    validate_dataset_name,
    model,
    clip_model,
    tokenizer,
    args=None,
    test_data=None,
):
    # 切换模型到评估模式，关闭dropout等行为
    model.eval()
    # 加载模板
    with open(os.path.join("./data", "templates.json")) as f:
        templates = json.load(f)[args.validate_dataset_prompt]

    # 加载分割提示
    with open(os.path.join("./data", "part_temple.json")) as f:
        part_data = json.load(f)

    part_temple = part_data[args.seg_cat]

    with torch.no_grad():
        logging.info("=> 编码分割提示")
        segmentation_texts_features = []  # 编码后的分割提示+模板
        # 每次只验证一类
        for p in part_temple:
            segmentation_texts = [t.format(p) for t in templates]
            segmentation_texts = tokenizer(segmentation_texts).to(
                device=args.device, non_blocking=True
            )  # 编码token，下一步通过clip编码

            # 确保文本是二维的
            if len(segmentation_texts.shape) < 2:
                segmentation_texts = [None, ...]

            # 使用clip编码一个分割提示
            clip_embeddings = clip_model.encode_text(segmentation_texts)  # [64, 1024]
            clip_embeddings = clip_embeddings / clip_embeddings.norm(
                dim=-1, keepdim=True
            )  # 归一化 [64, 1024]
            clip_embeddings = clip_embeddings.mean(dim=0)  # 求平均 [1024]
            clip_embeddings = clip_embeddings / clip_embeddings.norm(
                dim=-1, keepdim=True
            )  # 再次归一化[1024]
            segmentation_texts_features.append(clip_embeddings)

        # 将分类提示堆叠起来[K, 1024]
        segmentation_texts_features = torch.stack(segmentation_texts_features, dim=0)
        for batch_idx, (xyz, label_index, label_name, rgb) in enumerate(test_loder):
            # non_blocking 异步执行数据移动操作，需要配合 pin_memory=True 确保数据在锁业内存中
            xyz = xyz.to(device=args.device, non_blocking=True)
            rgb = rgb.to(device=args.device, non_blocking=True)
            feature = torch.cat((xyz, rgb), dim=-1)  # 连接点云位置和颜色信息

            # 编码点云
            # patch_features[B, G, 1024]
            # my_idx[B, G, M],
            # xyz[B, 10000, 3],
            # rgb[B, 10000, 3]
            patch_features, my_idx, cls_token = utils.get_model(model).encode_pc(
                feature
            )
            patch_features = patch_features / patch_features.norm(
                dim=-1, keepdim=True
            )  # [B, G, 1024] 每个patch的特征向量

            # 使用k-means 类聚检查patch_features的效果
            batch_labels = []
            batch_cluster_centers = []

            for i in range(patch_features.shape[0]):
                # 当前 batch 的特征
                current_batch_features = patch_features[i]  # [G, 1024]

                # 使用 kmeans-pytorch 聚类
                # 返回cluster_ids_x 分组信息[G]  cluster_centers中心点信息[K, 1024]
                cluster_ids_x, cluster_centers = kmeans(
                    X=current_batch_features,  # 输入特征
                    num_clusters=args.k_means_num,  # 聚类数
                    distance="euclidean",  # 使用欧几里得距离
                    device=xyz.device,  # 设备
                )

                # 保存当前 batch 的聚类结果
                batch_labels.append(cluster_ids_x)  # 聚类标签

                # batch_cluster_centers.append(cluster_centers)  # 聚类中心

            # 确保 batch_cluster_centers 是一个 tensor 类型
            # stack 将 [B, K, 1024] 的单个张量堆叠在一起
            # batch_cluster_centers = torch.stack(batch_cluster_centers).to(xyz.device)
            batch_labels = torch.stack(batch_labels).to(xyz.device)

            # # 计算每个patch与每个文本提示的相似性 得到[B, G, K]
            # similarity_patch_centers = (
            #     batch_cluster_centers.float() @ segmentation_texts_features.float().t()
            # )

            # # 计算每个patch与每个文本提示的相似性 得到[B, G, K]
            # similarity_patch_part = (
            #     patch_features.float() @ segmentation_texts_features.float().t()
            # )

            # 返回最大值的索引 argmax
            # 返回值则为 max [B, G]
            # patch_seg_result_value, patch_seg_result = torch.max(
            #     similarity_patch_part, dim=-1
            # )

            ##检查patch类聚效果##
            patch_seg_result = batch_labels
            ####
            # 创建一个[B, N, K]的全0张量 batch_size N个点 每个在K-means类上被分类的次数
            K = args.k_means_num
            B, N, _ = xyz.shape
            point_seg_result = torch.zeros(B, N, K, device=args.device)  # [B, 10000, K]

            # 优化后的逻辑：去除三重循环，使用张量操作实现
            # 将 patch_seg_result 映射到点的维度
            # patch_seg_result 的形状是 [B, G]，扩展后是 [B, G, M] 与 my_idx 形状一致
            point_part_ids = patch_seg_result.unsqueeze(-1).expand(
                -1, -1, my_idx.shape[-1]
            )  # 每个点的分类结果

            # 将 my_idx 展平为 [B, G * M]
            flat_idx = my_idx.view(B, -1)  # 点索引

            # 将 point_part_ids 展平为 [B, G * M]
            # 使用view内存不连续问题，报错
            # flat_part_ids = point_part_ids.view(B, -1)  # 每个点的分类结果展平
            flat_part_ids = point_part_ids.reshape(B, -1)
            # 构造 one-hot 表示，形状为 [B, G * M, K]
            one_hot_part_ids = torch.nn.functional.one_hot(flat_part_ids, K).float()

            # 使用 scatter_add_ 更新 point_seg_result
            # 以 flat_idx 为索引， 将第i个点的one-hot编码，累加到point_seg_result[b][i]上
            for b in range(B):
                point_seg_result[b].scatter_add_(
                    dim=0,  # 在点索引维度上聚合
                    index=flat_idx[b]  # 点的索引G * M
                    .unsqueeze(-1)  # 添加一个维度 [G * M, 1]
                    .expand(
                        -1, K
                    ),  # 点索引扩展到 [G * M, K] 不会增加新的操作  ps:extend用于列表；expand用于张量不创建新的数据，repeat会真实创建
                    src=one_hot_part_ids[b],  # 对应的 one-hot 编码 [B * G, K]
                )

            # 根据每个类别的分类次数，投票确定每个点该分为哪一类
            point_seg_result = point_seg_result.max(dim=-1)[1]  # [B, N, K]->[B, N]

            # 根据k-means的分类结果[B, N]，我们可以从feature[B, N, 6]中拆分出K个mini点云
            # [N, 6]->[k, Nk, 6]

            # 获得k-means后，每个大patch的点云数据
            big_path_feature = []
            # 遍历 batch
            for b in range(B):
                batch_big_path_feature = []  # 存放当前 batch 的分类结果
                for k in range(K):
                    # 找到当前 batch 中属于类别 k 的点的索引
                    indices = (point_seg_result[b] == k).nonzero(as_tuple=True)[0]
                    # 获取对应的点云数据
                    selected_points = feature[b, indices]  # [Nk, 6]
                    batch_big_path_feature.append(
                        selected_points
                    )  # 每一类是一个 Tensor，大小为 [Nk, 6]
                big_path_feature.append(batch_big_path_feature)  # [B, K, NK, 6]

            # 将k-means后的每个大patch点云 使用uni3d编码
            for b in range(B):
                k_means_patch_cls = []
                for k in range(K):
                    patch_point_feature = big_path_feature[b][k].unsqueeze(0)
                    _, _, cls_token = utils.get_model(model).encode_pc(
                        patch_point_feature
                    )
                    k_means_patch_cls.append(cls_token)
                k_means_patch_cls = torch.stack(k_means_patch_cls, dim=0)#[K, 1, 1024]
                k_means_patch_cls = k_means_patch_cls.squeeze(1)  # [K, 1024]

                # 每个点云得到[K, 1024] 个大patch的编码特征
                # 计算k-means后的每个大path的cls与文本的相似度
                similarity_k_means_text_features = (
                    k_means_patch_cls.float() @ segmentation_texts_features.float().t()
                )  # 这里会得到[K, C]
                value, index = torch.max(similarity_k_means_text_features, dim=-1)
                # 新的分类结果 point_seg_result[B, N]
                point_seg_result[b] = index[point_seg_result[b].cpu().numpy()]
            #############编码完成###############
            # 画图写入html
            xyz_cpu = xyz.cpu().numpy()
            point_seg_result_rgb = point_seg_result.cpu().numpy()
            for i in range(B):  # 遍历每个点云
                update_class_counter(label_name[i])
                save_point_cloud_html(
                    xyz_cpu[i, :, 0],
                    xyz_cpu[i, :, 1],
                    xyz_cpu[i, :, 2],
                    point_seg_result_rgb[i, :],
                    label_name[i],
                    class_counters[label_name[i]],
                    part_data,
                    args,
                    dataset_name="modelnet",
                )


def test_zeroshot_3d(args, model, clip_model):
    # 加载uni3d模型权重
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    logging.info("loaded checkpoint {}".format(args.ckpt_path))

    # 提取模型参数sd（state_dict）
    # 如果是以分布式训练的，那么开头为module，需要去掉[module.]前缀
    sd = checkpoint["module"]
    if not args.distributed and next(iter(sd.items()))[0].startswith("module"):
        sd = {k[len("module.") :]: v for k, v in sd.items()}

    # pytorch 内置方法，加载模型权重
    model.load_state_dict(sd)

    # 初始化文本编码器
    tokenizer = SimpleTokenizer()
    # 自定义的dataset负责处理数据，最重要的是__getitem__
    test_dataset = utils.get_dataset(None, tokenizer, args, "val")
    # dataloader只负责加载数据
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 打乱整个数据集
        num_workers=args.workers,
        pin_memory=True,  # 加载数据到锁页内存中，提高数据加载效率
        sampler=None,  # 自定义采样器，如shuffle为True时，则为随机采样
        drop_last=False,  # 保留最后一个批次
    )
    results_modelnet = test_zeroshot_3d_core(
        test_loader,
        args.validate_dataset_name,
        model,
        clip_model,
        tokenizer,
        args,
        "modelnet",
    )
    return results_modelnet


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


if __name__ == "__main__":
    main(sys.argv[1:])
