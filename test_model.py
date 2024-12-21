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

# import debugpy

# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
args, ds_init = parse_args(sys.argv[1:])
device = init_distributed_device(args)

feature = torch.rand(1300, 6)
print(feature.shape)
if len(feature.shape) < 3:
    feature = feature.unsqueeze(0)
print(feature.shape)
feature=feature.to(device)
# 得到Uni3d的实例
model = getattr(models, args.model)(args=args)
model.to(device)

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

# 切换模型到评估模式，关闭dropout等行为
model.eval()

patch_features, my_idx, cls_token = utils.get_model(model).encode_pc(feature)
