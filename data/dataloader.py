import torch
from torch.utils.data import DataLoader


class CustomDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        sampler=None,
        drop_last=False,
        **kwargs
    ):
        # 初始化父类
        super(CustomDataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=sampler,
            drop_last=drop_last,
            **kwargs,
        )

    def __iter__(self):
        # 获取父类的迭代器
        data_iter = super(CustomDataLoader, self).__iter__()

        # 自定义处理：过滤掉 None 数据
        for batch in data_iter:
            if batch is not None:
                yield batch
