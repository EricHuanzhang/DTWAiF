import torch
import torch.nn as nn
import numpy as np

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        elif mode == 'transform':
            x = self._normalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class HeterogeneousRevIN(nn.Module):
    """
    终极异构可逆实例归一化 (Heterogeneous RevIN)
    1. 动态常量探测：遇到死线自动 Bypass，保留异常偏置。
    2. 内置离散探测：全张量化的高效差分探测，无需外部输入掩码。
    3. 异构路由：离散通道执行 Min-Max 映射，连续通道执行 Std-Norm。
    """

    def __init__(self, num_features: int, discrete_threshold=20, eps=1e-5, affine=True, subtract_last=False,
                 constant_tol=1e-5):
        super(HeterogeneousRevIN, self).__init__()
        self.num_features = num_features
        self.discrete_threshold = discrete_threshold
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.constant_tol = constant_tol

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        if mode == 'norm' or mode == 'transform':
            if mode == 'norm':
                self._get_statistics(x)
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise NotImplementedError

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))

        # ----------------------------------------------
        # A. 连续变量统计量 (Mean & Stdev)
        # ----------------------------------------------
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1).detach()
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()

        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

        # ----------------------------------------------
        # B. 常数与极差探测 (Min, Max, PTP)
        # ----------------------------------------------
        self.min_val = torch.min(x, dim=dim2reduce[0], keepdim=True)[0].detach()
        self.max_val = torch.max(x, dim=dim2reduce[0], keepdim=True)[0].detach()
        self.ptp = (self.max_val - self.min_val).detach()

        # Shape: [Batch, 1, Channels]
        self.is_constant = (self.ptp <= self.constant_tol)

        # ----------------------------------------------
        # C. 极致高效的内建张量化离散通道探测
        # 利用 GPU 并行排序与错位差分，绝不使用慢速 for 循环
        # ----------------------------------------------
        B = x.shape[0]
        C = x.shape[-1]
        x_flat = x.view(B, -1, C)  # Shape: [Batch, SeqLen, Channels]

        # 抹去浮点误差的影响 (相当于 numpy 的 round(decimals=5))
        x_rounded = torch.round(x_flat * 1e5)

        # 沿时间步维度进行排序
        x_sorted, _ = torch.sort(x_rounded, dim=1)

        # 错位差分：只要相邻元素不同，就是新出现的一个 unique 值
        diffs = (x_sorted[:, 1:, :] != x_sorted[:, :-1, :])

        # 累加不同项的数量，再加上第1个元素，得到精确的 unique_count
        unique_counts = diffs.sum(dim=1).view(B, 1, C) + 1

        # Shape: [Batch, 1, Channels]，动态记录每个通道的离散状态
        self.is_discrete_mask = (unique_counts <= self.discrete_threshold)

    def _normalize(self, x):
        x_norm = x.clone()

        # 利用刚刚内建探测得出的 mask 进行状态切分
        mask_discrete = self.is_discrete_mask & (~self.is_constant)
        mask_continuous = (~self.is_discrete_mask) & (~self.is_constant)

        # 路由 1: 离散通道
        if mask_discrete.any():
            x_disc = (x - self.min_val) / (self.ptp + self.eps)
            x_norm = torch.where(mask_discrete, x_disc, x_norm)

        # 路由 2: 连续通道
        if mask_continuous.any():
            if self.subtract_last:
                x_cont = (x - self.last) / (self.stdev + self.eps)
            else:
                x_cont = (x - self.mean) / (self.stdev + self.eps)

            if self.affine:
                x_cont = x_cont * self.affine_weight.view(1, 1, -1) + self.affine_bias.view(1, 1, -1)

            x_norm = torch.where(mask_continuous, x_cont, x_norm)

        # 常量通道被 Bypass，保留 clone 的原始值
        return x_norm

    def _denormalize(self, x):
        x_denorm = x.clone()

        # 重建掩码映射
        mask_discrete = self.is_discrete_mask & (~self.is_constant)
        mask_continuous = (~self.is_discrete_mask) & (~self.is_constant)

        # 逆路由 1: 离散通道
        if mask_discrete.any():
            x_disc = x * (self.ptp + self.eps) + self.min_val
            x_denorm = torch.where(mask_discrete, x_disc, x_denorm)

        # 逆路由 2: 连续通道
        if mask_continuous.any():
            x_cont = x
            if self.affine:
                x_cont = (x_cont - self.affine_bias.view(1, 1, -1)) / (self.affine_weight.view(1, 1, -1) + self.eps)

            if self.subtract_last:
                x_cont = x_cont * (self.stdev + self.eps) + self.last
            else:
                x_cont = x_cont * (self.stdev + self.eps) + self.mean

            x_denorm = torch.where(mask_continuous, x_cont, x_denorm)

        return x_denorm