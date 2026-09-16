from torch import nn, einsum
from einops import rearrange
import math, torch
from ..utils.ch_discover_loss import DynamicalContrastiveLoss

# =========================================================
# 🌟 创新组件：Band-wise 严格隔离算子簇
# =========================================================

class BandwiseLayerNorm(nn.Module):
    """频带隔离归一化：防止全局均值/方差计算导致不同小波频带的量纲污染"""
    def __init__(self, sub_dims):
        super().__init__()
        self.sub_dims = sub_dims
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in sub_dims])

    def forward(self, x):
        # 按各频带预分配的尺寸进行严格切片，独立归一化后重新拼接
        chunks = torch.split(x, self.sub_dims, dim=-1)
        out_chunks = [norm(c) for norm, c in zip(self.norms, chunks)]
        return torch.cat(out_chunks, dim=-1)

class BandwiseLinear(nn.Module):
    """频带隔离线性映射：本质是一个严格的分块对角矩阵，跨频带权重绝对为0"""
    def __init__(self, sub_dims_in, sub_dims_out, bias=True):
        super().__init__()
        self.sub_dims_in = sub_dims_in
        self.sub_dims_out = sub_dims_out
        self.linears = nn.ModuleList([
            nn.Linear(d_in, d_out, bias=bias) for d_in, d_out in zip(sub_dims_in, sub_dims_out)
        ])

    def forward(self, x):
        chunks = torch.split(x, self.sub_dims_in, dim=-1)
        out_chunks = [linear(c) for linear, c in zip(self.linears, chunks)]
        return torch.cat(out_chunks, dim=-1)

class BandwiseFeedForward(nn.Module):
    """频带隔离前馈网络：每个频带在自己的子空间内独立升降维与非线性激活"""
    def __init__(self, sub_dims, mlp_dim, dropout=0.5):
        super().__init__()
        self.sub_dims = sub_dims
        total_dim = sum(sub_dims)
        self.nets = nn.ModuleList()
        for d in sub_dims:
            # 按照该频带的宽度比例分配隐藏层容量，保证总参数量与原 mlp_dim 一致
            d_hidden = max(1, int(mlp_dim * (d / total_dim)))
            self.nets.append(nn.Sequential(
                nn.Linear(d, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, d),
                nn.Dropout(dropout)
            ))

    def forward(self, x):
        chunks = torch.split(x, self.sub_dims, dim=-1)
        out_chunks = [net(c) for net, c in zip(self.nets, chunks)]
        return torch.cat(out_chunks, dim=-1)

# =========================================================
# 🌟 架构组件：包裹频带隔离算子的 Transformer 模块
# =========================================================

class DualPreNorm(nn.Module):
    """
    双轨预归一化层：分别为物理流(x_phys)和拓扑流(x_topo)提供独立的 LayerNorm
    """
    def __init__(self, dim, sub_dims, fn):
        super().__init__()
        # 物理重构轨 (x_phys) 启用频带隔离归一化
        self.norm_phys = BandwiseLayerNorm(sub_dims)
        # 拓扑寻星轨 (x_topo) 使用全局常规归一化
        self.norm_topo = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x_phys, x_topo, **kwargs):
        return self.fn(self.norm_phys(x_phys), self.norm_topo(x_topo), **kwargs)

class BandwisePreNorm(nn.Module):
    def __init__(self, sub_dims, fn):
        super().__init__()
        self.norm = BandwiseLayerNorm(sub_dims)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class c_Attention(nn.Module):
    def __init__(self, dim, sub_dims, heads, dim_head, dropout=0.8, regular_lambda=0.3, temperature=0.1):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.d_k = math.sqrt(self.dim_head)
        inner_dim = dim_head * heads
        self.attend = nn.Softmax(dim=-1)

        # 1. 拓扑空间 (Topology Space): 允许全局映射计算注意力相似度图谱
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)

        # 计算内部投影的隔离尺寸 (适配 inner_dim != dim 的情况)
        total_dim = sum(sub_dims)
        self.inner_sub_dims = [max(1, int(inner_dim * (d / total_dim))) for d in sub_dims]
        self.inner_sub_dims[0] += inner_dim - sum(self.inner_sub_dims)  # 补齐四舍五入的余数

        # 2. 物理空间 (Reconstruction Space): V 和 Out 保持绝对频带隔离！
        self.to_v = BandwiseLinear(sub_dims, self.inner_sub_dims)
        self.to_out = nn.Sequential(
            BandwiseLinear(self.inner_sub_dims, sub_dims),
            nn.Dropout(dropout)
        )
        self.dynamicalContranstiveLoss = DynamicalContrastiveLoss(k=regular_lambda, temperature=temperature)

    def forward(self, x_phys, x_topo, attn_mask=None):
        h = self.heads
        # =========================================================
        # 🌟 【优化 3 核心落地：Query-Value Decoupling】
        # =========================================================
        # 1. 拓扑图谱空间 (Topology Space): Q 和 K 仅由纯净宏观物理趋势 x_topo 生成
        q = self.to_q(x_topo)
        k = self.to_k(x_topo)

        # 2. 物理重构空间 (Reconstruction Space): V 由包含所有高频微观细节的 x_phys 生成
        v = self.to_v(x_phys)

        scale = 1 / self.d_k

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        dynamical_contrastive_loss = None

        scores = einsum('b h i d, b h j d -> b h i j', q, k)

        if attn_mask is not None:
            q_norm = torch.norm(q, dim=-1, keepdim=True)
            k_norm = torch.norm(k, dim=-1, keepdim=True)
            norm_matrix = torch.einsum('bhid,bhjd->bhij', q_norm, k_norm)
            def _mask(scores, attn_mask):
                # 1. 定义一个极小的负数 (近似负无穷)
                # large_negative = -math.log(1e10)
                large_negative = -1e9
                # 2. 构造加法掩码：
                # 如果 attn_mask 为 0 (无关)，则设为 large_negative
                # 如果 attn_mask 为 1 (相关)，则设为 0

                # 3. 核心切断操作 (乘法+加法 混合操作)：
                # 第一步 scores * attn_mask：
                #    将无关连接的原始分数直接清零 (防止原始分数很大导致加了负数还不够小)
                # 第二步 + attention_mask：
                #    将无关连接的位置加上极小负数 (确保 Softmax 后为 0)
                attention_mask = torch.where(attn_mask == 0, large_negative, 0.0)
                scores = scores * attn_mask.unsqueeze(1) + attention_mask.unsqueeze(1)

                return scores

            masked_scores = _mask(scores, attn_mask)
            # 🌟 DCL 动态对比损失的“隔离防火墙”
            # DCL 损失的物理意义是“图谱动态对比聚类”，它需要依赖原始的、未被掩码遮蔽的相似度分数来产生对比梯度（拉近正样本，推远负样本）
            # 注意：计算 DCL 的 scores 来源于 x_topo 衍生的 q, k。
            # 因此，DCL 拉近相似节点的反向传播梯度只会回传到 x_topo。
            # 负责高频重构的 Value 特征被彻底隔离保护，绝不会被“同质化”！
            dynamical_contrastive_loss = self.dynamicalContranstiveLoss(scores, attn_mask, norm_matrix)
        else:
            masked_scores = scores

        # 4. Softmax 归一化：
        # 由于无关连接的分数已经是极小负数，Softmax(负无穷) -> 0
        attn = self.attend(masked_scores * scale)  # 注意力分数 self.attend = nn.Softmax(dim=-1)
        # 纯净的拓扑权重 (attn) 乘上 包含私有微观细节的物理特征 (v)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), attn, dynamical_contrastive_loss

class c_Transformer(nn.Module):  ##Register the blocks into whole network
    def __init__(self, dim, sub_dims, depth, heads, dim_head, mlp_dim, dropout=0.8, regular_lambda=0.3,
                 temperature=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # 注入 sub_dims 使得双轨预归一化生效
                DualPreNorm(dim, sub_dims,
                            c_Attention(dim, sub_dims, heads=heads, dim_head=dim_head, dropout=dropout,
                                        regular_lambda=regular_lambda, temperature=temperature)),
                # FFN 层使用 Bandwise 结构
                BandwisePreNorm(sub_dims, BandwiseFeedForward(sub_dims, mlp_dim=mlp_dim, dropout=dropout))
            ]))

    def forward(self, x_phys, x_topo, attn_mask=None):
        total_loss = 0
        all_attentions = [] # 初始化列表存储每一层的权重

        for attn_layer, ff in self.layers:
            # 💡 【架构亮点】：每一层都透传静态的宏观拓扑先验 x_topo
            # Q 和 K 在每层都会基于同一宏观趋势学到不同的拓扑子空间，确保全局物理一致性
            x_n, current_attn, dcloss = attn_layer(x_phys, x_topo, attn_mask=attn_mask)

            # 收集当前层的注意力权重
            all_attentions.append(current_attn)

            if dcloss is not None:
                total_loss += dcloss

            # 💡 【残差隔离】：残差累加和 FFN 变换 仅发生在物理重构轨 (x_phys)
            # x_topo 不参与残差累加，彻底杜绝了深层网络的注意力拓扑发生“语义漂移 (Semantic Drift)”
            x_phys = x_n + x_phys
            x_phys = ff(x_phys) + x_phys

        #dcloss = total_loss / len(self.layers)
        # 确保返回的 dcloss 始终是合法的 Tensor
        if isinstance(total_loss, torch.Tensor):
            dcloss = total_loss / len(self.layers)
        else:
            dcloss = torch.tensor(0.0, device=x_phys.device)

        # 将列表堆叠为张量:
        stacked_attn = torch.stack(all_attentions, dim=1)

        return x_phys, stacked_attn, dcloss


class Trans_C(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head, dropout, patch_dim, horizon, d_model,
                 regular_lambda=0.3, temperature=0.1):
        super().__init__()

        self.dim = dim
        self.patch_dim = patch_dim
        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_dim, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.transformer = c_Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, regular_lambda=regular_lambda,
                                         temperature=temperature)

        self.mlp_head = nn.Linear(dim, d_model)  # horizon)

    def forward(self, x, attn_mask=None):
        x = self.to_patch_embedding(x)
        x, attn, dcloss = self.transformer(x, attn_mask)
        x = self.dropout(x)
        x = self.mlp_head(x).squeeze()
        return x, dcloss  # ,attn
