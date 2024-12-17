import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils

import logging


def fps(data, number):
    """
    data B N 3
    number int
    """
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = (
        pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx)
        .transpose(1, 2)
        .contiguous()
    )
    return fps_data


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token
        logging.info("patch dropout prob is {}".format(prob))

    def forward(self, x):
        # if not self.training or self.prob == 0.:
        #     return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz, color):
        batch_size, num_points, _ = xyz.shape
        # 中心点
        center = fps(xyz, self.num_group)  # [B, G, 3]
        idx = knn_point(self.group_size, xyz, center)  # [B, G, M]

        # 用于后面的投票机制
        my_idx = idx

        # 需要把 batch 中的数据展平，所以对于每一个点云的index，我们需要加上偏移量
        idx_base = (
            torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        )  # [B] -> [B, 1, 1]

        # 加上索引值
        idx = idx + idx_base

        # 多维数据使用index取值：都转化成一维数据
        # 将索引展平
        idx = idx.view(-1)
        # 将坐标展平
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(
            batch_size, self.num_group, self.group_size, 3
        ).contiguous()

        # 同样取颜色特征
        neighborhood_color = color.view(batch_size * num_points, -1)[idx, :]
        neighborhood_color = neighborhood_color.view(
            batch_size, self.num_group, self.group_size, 3
        ).contiguous()

        # 取中心化，unsqueeze是在2的位置插入一个维度就变成了[B, G, M, 3]-[B, G, 1, 3]
        neighborhood = neighborhood - center.unsqueeze(2)

        # 合并特征，位置和坐标
        features = torch.cat((neighborhood, neighborhood_color), dim=-1)
        return neighborhood, center, features, my_idx


class Encoder(nn.Module):

    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(6, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(
                inplace=True
            ),  # inplace=True 直接在张量上面修改；False 返回一个新的张量，原张量保持不变
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        """
        point_groups : B G N 3
        -----------------
        feature_global : B G C
        """
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 6)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1
        )  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class PointcloudEncoder(nn.Module):

    def __init__(self, point_transformer, args):
        super().__init__()

        self.trans_dim = args.pc_feat_dim  # 点云编码后的特征
        self.embed_dim = args.embed_dim  # clip编码后的维度
        self.group_size = args.group_size
        self.num_group = args.num_group

        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # 定义点云编码器
        self.encoder_dim = args.pc_encoder_dim
        self.encoder = Encoder(encoder_channel=self.encoder_dim)

        # 升维到ViT所需要的维度
        self.encoder2trans = nn.Linear(self.encoder_dim, self.trans_dim)

        # 经过ViT编码的升维，与clip编码后的text一致
        self.trans2embed = nn.Linear(self.trans_dim, self.embed_dim)

        # cls token nn.Parameter会将这个张量注册为模型的参数，并且可以随着模型更新
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        # 中心点位置嵌入模块3->768
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim)
        )

        self.patch_dropout = (
            PatchDropout(args.patch_dropout)
            if args.patch_dropout > 0.0
            else nn.Identity()  # 仅仅作为一个占位符，不对数据做任何改变
        )
        self.visual = point_transformer

    def forward(self, pts, colors):
        # 原始点云pts[10000, 3] ,color[10000, 3]

        # center[B, G, 3] features[B, G, M, 6]
        neighborhood, center, features, my_idx = self.group_divider(
            pts, colors
        )  # 定义在Group类

        # 把特征编码成[B, G, M]
        group_input_tokens = self.encoder(features)

        # 经过线形层，升维到ViT需要的维度
        group_input_tokens = self.encoder2trans(group_input_tokens)

        # 扩展cls_token，满足batchsize维度
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        # 对中心点位置编码
        pos = self.pos_embed(center)

        # 拼接cls_token
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        # 拼接位置编码
        pos = torch.cat((cls_pos, pos), dim=1)

        # ViT 完整的输入
        x = x + pos
        x0 = x[:, 1:, :]
        # ViT编码
        for i, blk in enumerate(self.visual.blocks):
            x = blk(x)
            if i + 1 == 4:
                x4 = x
            if i + 1 == 8:
                x8 = x

        # 提取出第四层和第八层
        x4 = self.visual.norm(x4[:, 1:, :])
        x8 = self.visual.norm(x8[:, 1:, :])
        # 使用patch features
        x12 = self.visual.norm(x[:, 1:, :])
        cls_token = self.visual.norm(x[:, 0, :])  # 添加全局信息
        cls_token = cls_token.unsqueeze(1).expand(-1, x.shape[1] - 1, -1)
        weights = torch.tensor([1.0, 1.0, 1.0, 100.0, 20.0])
        patch_features = (
            x0 * weights[0]
            + x4 * weights[1]
            + x8 * weights[2]
            + x12 * weights[3]
            + cls_token * weights[4]
        )
        patch_features = self.visual.fc_norm(patch_features)

        patch_features = self.trans2embed(patch_features)

        return patch_features, my_idx
