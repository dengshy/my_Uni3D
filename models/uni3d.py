import torch
import timm
import numpy as np

from torch import nn


class Uni3D(nn.Module):

    def __init__(self, point_encoder):
        super().__init__()

    def encode_pc(self, pc):
        pass

    def forward(self, pc, text, image):
        pass

def get_filter_loss(args):
    return
def get_metric_names(model):
    return
def create_uni3d(args):
    return
