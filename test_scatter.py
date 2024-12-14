import torch

output = torch.zeros(5, 3, dtype=torch.float)  # 初始化形状 [5, 3] 的目标张量
index = torch.tensor([1, 1, 1, 1])  # 指定位置索引，形状 [4]
src = torch.tensor(
    [[1, 2, 3], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=torch.float
)  # 数据，形状 [4, 3]
index_ = index.unsqueeze(-1).expand(-1, 3)
print(index_)
# scatter_add 在第 0 维操作，将 src 的每行加到 output 对应索引行上
# index[1]=[1,1,1] 将src[1]中的数据累加到output[1]中
# index[1]=[1,2,1] 将src[1]中的数据累加到output[1]和output[1]中
# 这就是为什么扩展没有影响的原因
output.scatter_add_(0, index.unsqueeze(-1).expand(-1, 3), src)
print(output)
