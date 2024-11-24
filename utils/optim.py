import torch
import logging
import re
import json
from .distributed import is_master

try:
    from apex.optimizers import FusedAdam
except:
    print("Please install lastest apex to use FusedAdam and FusedLAMB")
    FusedAdam, FusedLAMB = None, None


def get_num_layer_for_transformer(param_name, num_max_layer):
    return


class LayerDecayValueAssigner(object):
    pass


def get_parameters(args, model, assigner, tower):
    pass


def get_assigner(args, model):
    pass


def get_all_parameter(args, model):
    pass


def create_optimizer(args, model, return_params=False):
    pass

def get_loss_scale_for_deepspeed(model):
    pass

def get_grad_norm_():
    return