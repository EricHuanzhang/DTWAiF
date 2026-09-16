import torch
import torch.nn as nn
import ptwt  # 导入PyTorch可微小波变换工具箱
import pywt  # 用于获取标准小波基参数
import torch.nn.functional as F
from einops import rearrange
from typing import List

# 引用 CATCH 原有组件 (需确保路径正确)
from ts_benchmark.baselines.catch.layers.RevIN import RevIN,HeterogeneousRevIN
#from ts_benchmark.baselines.catch.layers.cross_channel_Transformer import c_Transformer
from ts_benchmark.baselines.catch.layers.cross_channel_bandwise_Transformer import BandwiseLayerNorm, c_Transformer
from ts_benchmark.baselines.catch.layers.cross_channel_Transformer import c_Transformer_original
from ts_benchmark.baselines.catch.layers.channel_mask import channel_mask_generator


# =========================================================================
# 🌟 消融实验替代组件：简单的 MLP 编码器与解码器
# =========================================================================
class SimpleMLPWaveletEncoder(nn.Module):
    """
    消融实验专用：简单的 MLP 编码器。
    将所有频带粗暴地拼接在一起，经过全连接层映射，抹杀掉多分支的频带隔离特性。
    """

    def __init__(self, d_model, coeff_shapes, dropout_rate=0.1):
        super(SimpleMLPWaveletEncoder, self).__init__()
        self.total_length = sum(coeff_shapes)

        self.mlp = nn.Sequential(
            nn.Linear(self.total_length, d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, coeffs_list):
        # 1. 粗暴拼接所有频带小波系数 [Batch, N_Vars, total_length]
        x_concat = torch.cat(coeffs_list, dim=-1)

        # 2. 混合映射
        x_embedded = self.mlp(x_concat)
        x_embedded = self.norm(x_embedded)

        # 3. 核心消融：返回两个相同的张量，彻底抹除 cA (拓扑轨) 和 cD (物理轨) 的隔离与解耦！
        # 让后续 Transformer 使用完全相同的特征去做 Query 和 Value。
        return x_embedded, x_embedded


class SimpleMLPWaveletDecoder(nn.Module):
    """
    消融实验专用：简单的 MLP 解码器。
    通过全连接层将隐变量直接映射到总长度，然后再强行切分回小波系数结构。
    """

    def __init__(self, d_model, coeff_shapes, dropout_rate=0.1):
        super(SimpleMLPWaveletDecoder, self).__init__()
        self.coeff_shapes = coeff_shapes
        self.total_length = sum(coeff_shapes)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, self.total_length)  # 直接映射出所有频带总长
        )

    def forward(self, latent_repr):
        # 1. 直接输出总长 [Batch, N_Vars, total_length]
        recon_concat = self.mlp(latent_repr)

        # 2. 按照原来的 coeff_shapes 强行切片，恢复成列表结构
        recon_coeffs_list = []
        cursor = 0
        for length in self.coeff_shapes:
            recon_coeffs_list.append(recon_concat[..., cursor: cursor + length])
            cursor += length

        return recon_coeffs_list


class ScaleAwareWaveletEncoder(nn.Module):
    """
    终极正交多分支小波编码器：
    1. 拓扑轨：独享全量 d_model，提供极度纯净、高维度的寻星锚点。
    2. 物理轨：切分 d_model 为多条平行车道，使用 Concatenation 替代 Addition，彻底消灭频带互相遮蔽的瓶颈。
    """

    def __init__(self, d_model, coeff_shapes, dropout_rate=0.1):
        super(ScaleAwareWaveletEncoder, self).__init__()
        num_bands = len(coeff_shapes)

        # =======================================================
        # 轨道一：拓扑轨 (仅用 cA 驱动，独享完整的 d_model 维度)
        # =======================================================
        self.topo_encoder = nn.Sequential(
            nn.Linear(coeff_shapes[0], d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.topo_norm = nn.LayerNorm(d_model)

        # =======================================================
        # 轨道二：物理轨 (绝对隔离的正交子空间分配)
        # =======================================================
        # 将 d_model 均匀切分为 num_bands 份
        self.sub_dims = [d_model // num_bands] * num_bands
        self.sub_dims[0] += d_model % num_bands  # 把除不尽的余数补偿给承载最多宏观信息的低频 cA

        self.phys_branches = nn.ModuleList()
        # ⚠️ 每个频带只被映射到属于它的专属子维度 (sub_dim)
        for length, sub_dim in zip(coeff_shapes, self.sub_dims):
            branch = nn.Sequential(
                nn.Linear(length, sub_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            self.phys_branches.append(branch)

        #self.fusion_norm = nn.LayerNorm(d_model)
        self.fusion_norm = BandwiseLayerNorm(self.sub_dims)

    def forward(self, coeffs_list):
        # 1. 生成纯净的宏观拓扑特征 (用于 Q, K)
        x_topo = self.topo_norm(self.topo_encoder(coeffs_list[0]))

        # 2. 生成绝对隔离的物理重构特征 (用于 V)
        phys_subs = []
        for coeff, branch in zip(coeffs_list, self.phys_branches):
            # 每个 branch 输出的 shape 为 [Batch, N_Vars, sub_dim]
            phys_subs.append(branch(coeff))

        # ⚠️ 核心质变：使用 cat 拼接，彻底抛弃加法！
        # 拼接后的总维度恰好严丝合缝等于 d_model
        # 此时的张量中，[0~31]维是cA，[32~63]维是cD_n... 物理结构极其明确
        x_phys_concat = torch.cat(phys_subs, dim=-1)
        x_phys = self.fusion_norm(x_phys_concat)

        return x_phys, x_topo


class ScaleAwareWaveletDecoder(nn.Module):
    """
    终极正交多分支小波解码器：
    不再从混合的 d_model 中盲目捞取数据。
    每个频带的解码头严格通过“张量切片 (Slicing)”读取专属子空间，
    完成 100% 物理隔离的高保真逆向还原。
    """

    def __init__(self, d_model, coeff_shapes, dropout_rate=0.1):
        super(ScaleAwareWaveletDecoder, self).__init__()
        num_bands = len(coeff_shapes)

        # 保持与 Encoder 完全一致的子空间切分逻辑
        self.sub_dims = [d_model // num_bands] * num_bands
        self.sub_dims[0] += d_model % num_bands

        self.branch_decoders = nn.ModuleList()
        for length, sub_dim in zip(coeff_shapes, self.sub_dims):
            # ⚠️ 注意：这里的输入维度是专属的 sub_dim，而不是臃肿的 d_model！
            branch = nn.Sequential(
                nn.Linear(sub_dim, sub_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(sub_dim, length)  # 输出无激活函数，包容真实小波系数的正负值
            )
            self.branch_decoders.append(branch)

    def forward(self, latent_repr):
        # latent_repr 是经过 Transformer 处理后的特征，Shape: [Batch, N_Vars, d_model]
        recon_coeffs_list = []
        cursor = 0

        # 遍历所有频带的分配尺寸和专属解码头
        for sub_dim, decoder in zip(self.sub_dims, self.branch_decoders):
            # ⚠️ 核心质变：精准张量切片！
            # 仅截取当前频带的专属子空间喂给解码头，杜绝任何跨频带的数值干扰
            band_latent = latent_repr[..., cursor: cursor + sub_dim]

            recon_coeffs_list.append(decoder(band_latent))
            cursor += sub_dim

        return recon_coeffs_list

class iCATCHModel(nn.Module):
    def __init__(self, configs, discrete_mask=None):
        super(iCATCHModel, self).__init__()

        # --- 基础配置 ---
        self.seq_len = configs.seq_len
        self.c_in = configs.c_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.ablation_QVDecoupling = configs.ablation_QVDecoupling

        # Top-K 稀疏率 (0.0~1.0, 1.0表示不稀疏)
        self.top_k_ratio = getattr(configs, 'top_k_ratio', 0.2)

        # --- 1. 数据预处理 ---
        #self.revin_layer = RevIN(self.c_in, affine=configs.affine, subtract_last=configs.subtract_last)
        # 获取探测到的离散特征掩码
        # self.revin_layer = HeterogeneousRevIN(
        #     num_features=self.c_in,
        #     eps=1e-5,
        #     affine=configs.affine
        # )
        # --- 2. 定义核心小波算子参数 ---
        # 'db4' (Daubechies 4) 因其具有较好的紧支撑性与平滑度，非常适合处理带有阶跃与局部奇异点的工业信号
        self.wavelet = pywt.Wavelet('db4')
        # 自动计算在给定序列长度下允许的最大小波分解层数
        self.level = pywt.dwt_max_level(self.seq_len, self.wavelet.dec_len)

        # --- 3. 倒置通道融合模块 (iTransformer Innovation) ---
        # 复用 CATCH 的掩码生成器，但输入维度调整为全局特征维度
        # Mask Generator 输入通常是原始特征的某种 concat，这里我们用 MLP 融合后的特征
        self.mask_generator = channel_mask_generator(input_size=self.d_model, n_vars=self.c_in)

        # --- 4. 预先计算一次DWT分解后所有尺度系数在时间轴上的长度总和 ---
        # 这是为了构建特征维度精确的线性映射矩阵
        dummy_input = torch.zeros(1, 1, self.seq_len)
        # 使用 'symmetric' 对齐正向传播，预计算形状
        dummy_coeffs = ptwt.wavedec(dummy_input, self.wavelet, level=self.level, mode='symmetric')
        self.coeff_shapes = [coeff.shape[-1] for coeff in dummy_coeffs]

        # --- 5. 实例化小波域特征提取感知机 ---
        # ⚠️ 修复一：启用多分支编码器，彻底删除单层 WaveletDomainMLP 和 _pack_coefficients
        if not self.ablation_QVDecoupling:
        #if getattr(configs, 'open', True):
            self.wave_encoder = ScaleAwareWaveletEncoder(self.d_model, self.coeff_shapes, configs.dropout)
        else:
            self.wave_encoder = SimpleMLPWaveletEncoder(self.d_model, self.coeff_shapes, configs.dropout)
        # --- 6. Transformer 核心 (c_Transformer) ---
        # 注意：这里不再需要 Trans_C 里的 Patch Embedding，因为我们已经做完了
        if not self.ablation_QVDecoupling:
            self.encoder = c_Transformer(
                dim=configs.d_model,
                sub_dims=self.wave_encoder.sub_dims,  # 🌟 关键：透传频带切片尺寸
                depth=configs.e_layers,
                heads=configs.n_heads,
                dim_head=configs.head_dim, #head_dim
                mlp_dim=configs.d_ff,       #d_ff
                dropout=configs.dropout,
                regular_lambda=configs.regular_lambda,
                temperature=configs.temperature
            )
        else:
            self.encoder = c_Transformer_original(
                dim=configs.d_model,
                depth=configs.e_layers,
                heads=configs.n_heads,
                dim_head=configs.head_dim,  # head_dim
                mlp_dim=configs.d_ff,  # d_ff
                dropout=configs.dropout,
                regular_lambda=configs.regular_lambda,
                temperature=configs.temperature
            )

        # --- 7. 重构投影 ---
        # 将Transformer输出的高阶隐空间表征重新投影回小波系数维度,启用尺度感知多分支解码体系
        if not self.ablation_QVDecoupling:
        #if getattr(configs, 'open', True):
            self.reconstruction_projector = ScaleAwareWaveletDecoder(
                d_model=self.d_model,
                coeff_shapes=self.coeff_shapes,
                dropout_rate=configs.dropout
            )
        else:
            self.reconstruction_projector = SimpleMLPWaveletDecoder(
                d_model=self.d_model,
                coeff_shapes=self.coeff_shapes,
                dropout_rate=configs.dropout
            )

    def forward(self, x):
        # x: [Batch, Seq_Len, N_Vars]

        # 步骤 1: 归一化
        #x_norm = self.revin_layer(x, 'norm')
        # 步骤 2: 张量轴置换以适配批量并行一维小波变换
        #x_permuted = x_norm.permute(0, 2, 1)  # -> [B, N, T]
        x_permuted = x.permute(0, 2, 1)  # -> [B, N, T]

        # 步骤 3: 核心算子 - 前端一维离散小波变换 (1D DWT)
        # 使用 mode='zero' (零填充) 配合 ptwt 内置算法处理边界问题
        wavelet_coeffs_list = ptwt.wavedec(x_permuted, self.wavelet, level=self.level, mode='symmetric')

        # 步骤 4: 多分支独立提特征
        #wave_embedding = self.wave_encoder(wavelet_coeffs_list)
        # 🌟 获取双轨特征
        fused_embedding, cA_topology_embedding = self.wave_encoder(wavelet_coeffs_list)
        # 步骤 5: 生成跨通道动态注意力掩码矩阵
        # 🌟 核心落地：cA-Driven Topology
        # 掩码生成器仅接收无高频噪声的 cA_topology_embedding！
        # 这将彻底解决高频热噪声引发的关联图谱闪烁 (Flickering)，连通度极度稳定。
        #channel_mask = self.mask_generator(cA_topology_embedding)
        # 🌟 【终极优化】：如果是极低维数据 (<=3 维，如 CalIt2, NYC)，
        # 强制废弃稀疏图谱，传入 None 以激活 Transformer 原生 Dense Attention，保证特征全量耦合！

        # 消融实验拦截逻辑：如果开启消融，强迫模型使用饱含高频噪声的物理融合张量来生成拓扑图谱和 Query/Key
        if not self.ablation_QVDecoupling:
            if self.c_in <= 1:
                channel_mask = None
            else:
                channel_mask = self.mask_generator(cA_topology_embedding)

            # 步骤 6: 小波域内执行跨通道特征交互聚合 (Masked Cross-Channel Attention)
            # Transformer依据掩码仅聚合同一物理因果簇内的特征变量
            # 将 fused_embedding 和 cA_topology_embedding 同时喂给 Transformer
            latent_repr, attn_weights, dcloss = self.encoder(
                x_phys=fused_embedding,         # 物理轨 (送入 Value)
                x_topo=cA_topology_embedding,   # 拓扑轨 (送入 Query/Key)
                attn_mask=channel_mask
            )
        else:
            if self.c_in <= 1:
                channel_mask = None
            else:
                channel_mask = self.mask_generator(fused_embedding)
            latent_repr, attn_weights, dcloss = self.encoder(fused_embedding,channel_mask)

        # 步骤 7: 结构化投影：解码器一步到位，直接输出 [cA, cD_n, ..., cD_1] 列表结构
        recon_coeffs_list = self.reconstruction_projector(latent_repr)

        # 步骤 8: 核心算子 - 终端一维逆离散小波变换 (1D IDWT) 直接在时域合成无伪影信号,列表直接馈入逆离散小波变换 (IDWT)
        out_reconstructed = ptwt.waverec(recon_coeffs_list, self.wavelet)

        # 步骤 9: 序列边缘裁剪以对齐原始序列长度 (处理由填充引起的微小长度溢出)
        # if out_reconstructed.shape[-1] > self.seq_len:
        #     out_reconstructed = out_reconstructed[..., :self.seq_len]
        if out_reconstructed.shape[-1] > self.seq_len:
            # 安全裁剪：去掉两端多余的部分（对称移除）
            excess = out_reconstructed.shape[-1] - self.seq_len
            start = excess // 2
            end = start + self.seq_len
            out_reconstructed = out_reconstructed[..., start:end]

        # # 步骤 10: 张量维度重置与统计量反归一化还原
        out_reconstructed = out_reconstructed.permute(0, 2, 1)  # 还原为


        #out_final = self.revin_layer(out_reconstructed, 'denorm')
        out_final = out_reconstructed

        # 返回主重构结果，附带对比损失(dcloss)、注意力图谱(用于可解释性分析)及预测小波系数(用于辅助损失计算)
        return out_final, recon_coeffs_list, dcloss, attn_weights, channel_mask

    def reconstruct_from_scaled(self, x_scaled, channel_drop_mask=None, fill_values=None):
        """
        C4 反事实重构接口（不依赖 RevIN）
        x_scaled: [B, T, N] —— 已是 HeterogeneousScaler 的 scaled space
        channel_drop_mask:
            - None：正常重构
            - [N] 或 [B,N] bool：True 表示遮蔽该通道
        fill_values:
            - [N] float tensor：每通道健康填充值（建议训练集 scaled 的 per-channel median）
        返回：out_final, recon_coeffs_list, dcloss, attn_weights, channel_mask
        """
        if channel_drop_mask is not None:
            if channel_drop_mask.dtype != torch.bool:
                channel_drop_mask = channel_drop_mask.bool()
            if channel_drop_mask.dim() == 1:
                channel_drop_mask = channel_drop_mask.unsqueeze(0).expand(x_scaled.size(0), -1)

            if fill_values is None:
                raise ValueError("C4 反事实遮蔽需要 fill_values（建议为训练集 scaled 的 per-channel median）")

            fv = fill_values.view(1, 1, -1).expand(x_scaled.size(0), x_scaled.size(1), -1)
            x_scaled = torch.where(channel_drop_mask.unsqueeze(1), fv, x_scaled)

        # 复用 forward 主链路
        x_permuted = x_scaled.permute(0, 2, 1)  # [B, N, T]
        wavelet_coeffs_list = ptwt.wavedec(x_permuted, self.wavelet, level=self.level, mode='symmetric')

        fused_embedding, cA_topology_embedding = self.wave_encoder(wavelet_coeffs_list)
        channel_mask = self.mask_generator(cA_topology_embedding)

        latent_repr, attn_weights, dcloss = self.encoder(
            x_phys=fused_embedding,
            x_topo=cA_topology_embedding,
            attn_mask=channel_mask
        )

        recon_coeffs_list = self.reconstruction_projector(latent_repr)

        out_reconstructed = ptwt.waverec(recon_coeffs_list, self.wavelet)
        if out_reconstructed.shape[-1] > self.seq_len:
            out_reconstructed = out_reconstructed[..., :self.seq_len]

        out_final = out_reconstructed.permute(0, 2, 1)  # [B, T, N]
        return out_final, recon_coeffs_list, dcloss, attn_weights, channel_mask
