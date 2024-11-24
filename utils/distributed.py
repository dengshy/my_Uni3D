import os
import json
import torch


def create_deepspeed_config(args):
    return


def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def prinit_rank_0():

    return


def is_dist_avail_and_initialized():
    return


def get_world_size():
    return


def is_using_distributed():
    return


def world_info_from_env():  # 环境变量中获取分布式训练信息的函数
    # 默认值：设置 local_rank 为 0，表示当前进程的本地 rank（用于单机多卡环境）。
    local_rank = 0
    # 检查环境变量中本地 rank 的信息，遍历常见的变量名称。
    for v in (
        "LOCAL_RANK",  # PyTorch 分布式训练中本地进程的 rank
        "MPI_LOCALRANKID",  # MPI 的本地 rank 变量
        "SLURM_LOCALID",  # SLURM 作业调度系统的本地 rank 变量
        "OMPI_COMM_WORLD_LOCAL_RANK",  # Open MPI 的本地 rank 变量
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])  # 获取并解析为整数
            break

    # 设置 global_rank 为 0，表示当前进程的全局rank。
    global_rank = 0

    # 检查环境变量中全局 rank 的信息，遍历常见的变量名称。
    for v in (
        "RANK",  # PyTorch 分布式训练中的全局 rank
        "PMI_RANK",  # PMI 的全局 rank 变量
        "SLURM_PROCID",  # SLURM 作业调度系统的全局 rank 变量
        "OMPI_COMM_WORLD_RANK",  # Open MPI 的全局 rank 变量
    ):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break

    # 表示全局的总进程数
    world_size = 1

    # 检查环境变量中 world size 的信息，遍历常见的变量名称。
    for v in (
        "WORLD_SIZE",  # PyTorch 分布式训练的总进程数
        "PMI_SIZE",  # PMI 的总进程数变量
        "SLURM_NTASKS",  # SLURM 作业调度系统的任务总数变量
        "OMPI_COMM_WORLD_SIZE",  # Open MPI 的总进程数变量
    ):
        if v in os.environ:
            world_size = int(os.environ[v])
            break
    return local_rank, global_rank, world_size


def init_distributed_device():
    return


def create_deepspeed_config():
    return
