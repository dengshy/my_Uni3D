import json
import os
import logging
from utils.params import parse_args
args, ds_init = parse_args(None)
# 加载模板
with open(os.path.join("./data", "templates.json")) as f:
    templates = json.load(f)[args.validate_dataset_prompt]

# 加载标签
with open(os.path.join("./data", "labels.json")) as f:
    labels = json.load(f)[args.validate_dataset_name]

# 加载分割提示
with open(os.path.join("./data", "part_temple.json")) as f:
    part_data = json.load(f)

print(labels)
print(part_data)
part_temple = {}  # 分割提示
for label in labels:
    if label in part_data:
        part_temple[label] = part_data[label]  # 按标签取出分割提示
    # else:
    #     logging.warning(f"没有{label}的分割提示模板，请添加")
    #     part_temple["label"] = ""
print(part_temple)