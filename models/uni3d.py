import torch
import timm
import numpy as np

from torch import nn
from . import losses
from .point_encoder import PointcloudEncoder


class Uni3D(nn.Module):

    def __init__(self, point_encoder):
        super().__init__()
        # 对比学习参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.point_encoder = point_encoder

    def encode_pc(self, pc):
        # 确保张量是连续储存的
        xyz = pc[:, :, :3].contiguous()
        color = pc[:, :, 3:].contiguous()

        # 模型验证调用此方法返回编码后的点云特征
        patch_features, my_idx = self.point_encoder(xyz, color)
        return patch_features, my_idx

    # 验证不需要用到forward
    def forward(self, pc, text, image):
        pass


def get_metric_names(model):
    return ["loss", "uni3d_loss", "pc_image_acc", "pc_text_acc"]


def get_filter_loss(args):
    return losses.Uni3d_Text_Image_Loss()


def create_uni3d(args):
    # 根据参数创建模型，并且加载预训练参数
    point_transformer = timm.create_model(
        args.pc_model,
        checkpoint_path=args.pretrained_pc,
        drop_path_rate=args.drop_path_rate,
    )
    point_encoder = PointcloudEncoder(point_transformer, args)

    model = Uni3D(
        point_encoder=point_encoder,
    )
    return model
