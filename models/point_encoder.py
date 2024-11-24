import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils

import logging

# ddd
def fps():
    return

def knn_point():
    return

def square_distance():
    return

class PatchDropout(nn.Module):
    pass

class Group(nn.Module):
    pass

class Encoder(nn.Module):

    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel

    def forward(self, point_groups):
        return


class PointcloudEncoder(nn.Module):

    def __init__(self, point_transformer, args):
        super().__init__()

    def forward(self, pts, colors):
        return
