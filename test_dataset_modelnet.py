from utils.tokenizer import SimpleTokenizer
from utils import utils
from utils.params import parse_args
import sys
import torch.utils.data
from torch.utils.data import DataLoader
from data.dataloader import CustomDataLoader
import os
import matplotlib.pyplot as plt  # 导入matplotlib库用于可视化
from utils.draw import save_point_cloud_html
import numpy as np


# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
def main(args):
    args, ds_init = parse_args(args)
    # 初始化文本编码器
    tokenizer = SimpleTokenizer()
    # 自定义的dataset负责处理数据，最重要的是__getitem__
    test_dataset = utils.get_dataset(None, tokenizer, args, "val")
    # dataloader只负责加载数据

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 打乱整个数据集
        num_workers=args.workers,
        pin_memory=True,  # 加载数据到锁页内存中，提高数据加载效率
        sampler=None,  # 自定义采样器，如shuffle为True时，则为随机采样
        drop_last=False,  # 保留最后一个批次
    )
    # 遍历 test_loader
    for batch_idx, (xyz, label, label_name, rgb) in enumerate(test_loader):
        print(f"Batch {batch_idx + 1}:")
        print(xyz.shape)  # 打印当前批次点云的形状，应该是(batch_size, N, 3)
        print(label)  # 打印当前批次的标签
        print(label_name)  # 打印当前批次的标签名称
        print(rgb.shape)  # 打印当前批次RGB信息的形状
        print("-" * 50)  # 打印分隔符，便于查看每个批次的内容
        weights = np.array([0.2989, 0.5870, 0.1140])  # 灰度加权系数
        gray_scale = np.dot(rgb, weights)  # [B, 10000]
        print(gray_scale.shape)
        # 假设 xyz 的形状是 (batch_size, N, 3)，即每个批次有 `batch_size` 个点云数据
        # 对于每个点云，生成一个3D可视化
        for i in range(xyz.shape[0]):  # 遍历每个点云
            save_point_cloud_html(
                xyz[i, :, 0],
                xyz[i, :, 1],
                xyz[i, :, 2],
                gray_scale[i, :],
                args.seg_cat,
                batch_idx * 8 + i + 1,
                output_directory="test_html",
                dataset_name="modelnet",
            )


if __name__ == "__main__":
    main(sys.argv[1:])
