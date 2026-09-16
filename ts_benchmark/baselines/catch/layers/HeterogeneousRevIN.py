import numpy as np
import torch
import torch.nn as nn
import platform
import os

# ==========================================
# 2. Attachment Class: HeterogeneousRevIN (Upgraded Network Layer)
# ==========================================
class HeterogeneousRevIN(nn.Module):
    """
    终极异构可逆实例归一化 (Heterogeneous RevIN)
    1. 动态常量探测 (Dynamic Constant Bypass)：遇到死线自动 Bypass，保留异常偏置。
    2. 异构路由 (Heterogeneous Routing)：离散通道执行 Min-Max 映射，连续通道执行 Std-Norm。
    """

    def __init__(self, num_features: int, discrete_mask=None, eps=1e-5, affine=True, subtract_last=False,
                 constant_tol=1e-5):
        super(HeterogeneousRevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.constant_tol = constant_tol

        if discrete_mask is not None:
            self.register_buffer('discrete_mask', torch.tensor(discrete_mask, dtype=torch.bool).view(1, 1, -1))
        else:
            self.register_buffer('discrete_mask', torch.zeros(1, 1, num_features, dtype=torch.bool))

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

        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1).detach()
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()

        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

        # 计算极差用于死线探测
        self.min_val = torch.min(x, dim=dim2reduce[0], keepdim=True)[0].detach()
        self.max_val = torch.max(x, dim=dim2reduce[0], keepdim=True)[0].detach()
        self.ptp = (self.max_val - self.min_val).detach()

        self.is_constant = (self.ptp <= self.constant_tol)

    def _normalize(self, x):
        x_norm = x.clone()
        mask_discrete = self.discrete_mask & (~self.is_constant)
        mask_continuous = (~self.discrete_mask) & (~self.is_constant)

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
        mask_discrete = self.discrete_mask & (~self.is_constant)
        mask_continuous = (~self.discrete_mask) & (~self.is_constant)

        if mask_discrete.any():
            x_disc = x * (self.ptp + self.eps) + self.min_val
            x_denorm = torch.where(mask_discrete, x_disc, x_denorm)

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