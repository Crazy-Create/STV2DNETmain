from torch.nn.parallel import DataParallel
import torch
from torch.nn.parallel._functions import Scatter
from torch.nn.parallel.parallel_apply import parallel_apply
# 这个文件的作用是多个计算并行进行
# 将输入的数据（如张量、元组、列表、字典）按给定的大小分配到多个 GPU 上，并返回分割后的数据。
def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    def scatter_map(obj):  # 递归的辅助函数，用于处理不同类型的对象
        if isinstance(obj, torch.Tensor): # 如果对象是Tensor类型，它会使用 Scatter.apply 来将数据分配到目标 GPU 上
            try:
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except:
                print('obj', obj.size())
                print('dim', dim)
                print('chunk_sizes', chunk_sizes)
                quit()
        if isinstance(obj, tuple) and len(obj) > 0: # 如果对象是元组、列表或字典，递归地调用 scatter_map，并对这些元素进行切分
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:  # 对于非张量类型的对象，直接返回目标设备的复制版本
        return scatter_map(inputs)
    finally:
        scatter_map = None

def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0): # 扩展了 scatter，支持处理 kwargs 字典
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []
    if len(inputs) < len(kwargs): # 如果 inputs 和 kwargs 长度不一致，扩展较短的一方，确保它们的长度相等
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs) # 转换为元组
    return inputs, kwargs


class BalancedDataParallel(DataParallel): # 继承自 DataParallel，在构造函数中新增了一个 gpu0_bsz 参数，用于设置 GPU 0 的 batch size
    def __init__(self, gpu0_bsz, *args, **kwargs):
        self.gpu0_bsz = gpu0_bsz
        super().__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids: # 如果没有指定设备 ID，则直接执行模型的前向传播
            return self.module(*inputs, **kwargs)

        if len(self.device_ids) == 1:
            inputs, kwargs = super().scatter(inputs, kwargs, self.device_ids)
            return self.module(*inputs[0], **kwargs[0])

        if self.gpu0_bsz == 0:
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids

        inputs, kwargs = self.scatter(inputs, kwargs, device_ids)

        if self.gpu0_bsz == 0:
            replicas = self.replicate(self.module, self.device_ids)
        else:
            replicas = self.replicate(self.module, self.device_ids[:len(inputs)])

        # replicas = self.replicate(self.module, device_ids[:len(inputs)])
        if self.gpu0_bsz == 0:
            replicas = replicas[1:]

        outputs = self.parallel_apply(replicas, device_ids, inputs, kwargs)
        return self.gather(outputs, self.output_device)
    # 函数将模型副本应用到输入数据上，并将结果返回。device_ids[:len(inputs)] 确保只使用与输入数量一致的 GPU
    def parallel_apply(self, replicas, device_ids, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, device_ids[:len(inputs)])

    def scatter(self, inputs, kwargs, device_ids):
        bsz = inputs[0].size(self.dim)
        num_dev = len(self.device_ids)
        gpu0_bsz = self.gpu0_bsz
        bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)

        if gpu0_bsz < bsz_unit:
            chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)
            delta = bsz - sum(chunk_sizes)
            for i in range(delta):
                chunk_sizes[i + 1] += 1
            if gpu0_bsz == 0:
                chunk_sizes = chunk_sizes[1:]
        else:
            return super().scatter(inputs, kwargs, device_ids)

        return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)