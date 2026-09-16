
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import math
import ptwt

class LogCoshLoss(nn.Module):
    """
    通用型基础 Log-Cosh 算子 (时域与小波域的底层物理引擎)
    """

    def __init__(self, reduction='mean'):
        super(LogCoshLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean'], "reduction 必须是 'none', 'sum' 或 'mean'"
        self.reduction = reduction
        self.log2 = math.log(2.0)

    def forward(self, pred, target):
        abs_error = torch.abs(pred - target)
        # 数值稳定的 Log-Cosh 实现
        loss = abs_error + F.softplus(-2.0 * abs_error) - self.log2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# 双域同构与“索伯列夫一阶差分”
class TimeDomainSobolevLoss(nn.Module):
    """
    时域专属：Log-Cosh 幅值约束 + 一阶差分(Sobolev)动力学约束
    完美支持 'none', 'mean', 'sum' 的 reduction 策略，防止时间轴张量广播崩溃。
    """
    """
    alpha: 一阶差分（速度）约束权重
    beta:  二阶差分（加速度/结构）约束权重
    """
    def __init__(self, alpha=0.5, beta=0.2, reduction='mean', base_criterion=None):
        super(TimeDomainSobolevLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean'], "reduction 必须是 'none', 'sum' 或 'mean'"
        self.reduction = reduction
        self.alpha = alpha
        self.beta = beta
        # ⚠️ 架构核心：内部基础算子强制设为 'none'，保留最大维度的时空矩阵
        if base_criterion is None:
            self.base_criterion = LogCoshLoss(reduction='none')
        else:
            self.base_criterion = base_criterion
            self.base_criterion.reduction = 'none'

    def forward(self, pred, target):
        # 1. 细粒度幅值损失矩阵，Shape: [Batch, Seq_Len, N_Vars]
        loss_amp = self.base_criterion(pred, target)

        total_loss = loss_amp
        seq_len = pred.shape[1]
        # 2. 一阶差分损失计算与时间轴对齐
        if self.alpha > 0 and seq_len > 1:
            diff_1_pred = pred[:, 1:, :] - pred[:, :-1, :]
            diff_1_target = target[:, 1:, :] - target[:, :-1, :]
            # Shape: [Batch, Seq_Len - 1, N_Vars]
            loss_diff_1 = self.base_criterion(diff_1_pred, diff_1_target)

            # ⚠️ 【核心修复】：为第0个时间步补充全 0 张量，完成时间轴对齐
            # F.pad 的参数顺序是从后向前的：(最后维度的左边, 最后维度的右边, 倒数第二维的左边, 倒数第二维的右边)
            # (0, 0, 1, 0) 意味着在 N_Vars (最后维度) 不补，在 Seq_Len (倒数第二维) 的头部补 1 个零，尾部不补
            loss_diff_1_padded = F.pad(loss_diff_1, (0, 0, 1, 0), mode='constant', value=0.0)
            # 此时两个张量维度严丝合缝 (均为 [Batch, Seq_Len, N_Vars])，完美相加
            total_loss = total_loss + self.alpha * loss_diff_1_padded

        if seq_len > 2 and self.beta > 0:
            # 3. 二阶差分约束 (Acceleration / 核心结构感知识别器)
            # 这将迫使模型学习信号的曲率。当测试集遇到"直线异常"时，
            # 真实信号二阶差分为0，而模型尝试重构正常波浪，二阶差分差距会极大。
            diff_2_pred = diff_1_pred[:, 1:, :] - diff_1_pred[:, :-1, :]
            diff_2_target = diff_1_target[:, 1:, :] - diff_1_target[:, :-1, :]
            loss_diff_2 = self.base_criterion(diff_2_pred, diff_2_target)
            loss_diff_2_padded = F.pad(loss_diff_2, (0, 0, 2, 0), mode='constant', value=0.0)
            total_loss = total_loss + self.beta * loss_diff_2_padded

        # 3. 统一执行外部指定的 Reduction 策略 (延迟规约)
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            # 返回完整的 [Batch, Seq_Len, N_Vars] 误差张量矩阵
            return total_loss

class WaveletDomainAuxiliaryLoss(nn.Module):
    """
    小波域专属：多尺度频带加权 Log-Cosh 损失函数
    """

    def __init__(self, configs, reduction='mean',base_criterion=None):
        super(WaveletDomainAuxiliaryLoss, self).__init__()
        self.reduction = reduction
        # 内部直接复用基础 LogCoshLoss 算子，设为 none 以便做逐元素加权
        if base_criterion is None:
            self.base_criterion = LogCoshLoss(reduction='none')
        else:
            self.base_criterion = base_criterion
            self.base_criterion.reduction = 'none'

    def forward(self, pred_coeffs_list, target_coeffs_list):
        weighted_losses = []
        # 1) cA (低频近似系数) 全量惩罚
        loss_cA_elem = self.base_criterion(pred_coeffs_list[0], target_coeffs_list[0])
        weighted_losses.append(loss_cA_elem)

        # 2) cD (高频细节系数) 频带指数衰减
        for i, (p_cD, t_cD) in enumerate(zip(pred_coeffs_list[1:], target_coeffs_list[1:])):
            weight = 1.0 / (2 ** (i + 1))
            loss_cD_elem = weight * self.base_criterion(p_cD, t_cD)
            weighted_losses.append(loss_cD_elem)

        # 3) 张量空间重组与 Reduction
        packed_weighted_loss = torch.cat(weighted_losses, dim=-1)

        if self.reduction == 'mean':
            return packed_weighted_loss.mean()
        elif self.reduction == 'sum':
            return packed_weighted_loss.sum()
        else:
            return packed_weighted_loss

class InverseTransformProjectionLoss(nn.Module):
    """
    逆变换投影误差：将频带误差通过 IDWT 投影回时域，得到与原始信号同维度的误差矩阵。
    用于在异常评分中引入频域信息，弥补时域误差在异常持续期间快速下降的缺陷。
    """
    def __init__(self, wavelet, level, seq_len, base_criterion=None, reduction='none'):
        """
        :param wavelet: pywt.Wavelet 对象，与模型使用的小波一致
        :param level: 小波分解层数
        :param seq_len: 原始序列长度，用于裁剪
        :param base_criterion: 逐点误差计算器，默认使用 LogCoshLoss(reduction='none')
        :param reduction: 返回结果的规约方式，'none' 返回逐点误差矩阵 [B, N, T]
        """
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.seq_len = seq_len
        if base_criterion is None:
            self.base_criterion = LogCoshLoss(reduction='none')
        else:
            self.base_criterion = base_criterion
        self.reduction = reduction

    def forward(self, pred_coeffs_list, target_coeffs_list):
        """
        pred_coeffs_list, target_coeffs_list: 每个元素为 [Batch, N_Vars, L_i]
        返回: 时域误差张量，形状 [Batch, N_Vars, seq_len]（若 reduction='none'）
        """
        # 1. 计算每个频带的逐点误差
        error_coeffs_list = []
        for i,(p, t) in enumerate(zip(pred_coeffs_list, target_coeffs_list)):
            #weight = 1.0 if i == 0 else 1.0 / (2 ** i)
            #err = self.base_criterion(p, t) * weight        # [B, N, L_i]
            err = self.base_criterion(p, t)  # [B, N, L_i]
            error_coeffs_list.append(err)

        # 2. 对误差列表进行逆小波变换，投影回时域
        time_error = ptwt.waverec(error_coeffs_list, self.wavelet)  # [B, N, T_out]

        # 3. 裁剪至原始序列长度
        if time_error.shape[-1] > self.seq_len:
            time_error = time_error[..., :self.seq_len]

        final_error = time_error.permute(0, 2, 1)    # [B, T, N]

        if self.reduction == 'mean':
            return final_error.mean()
        elif self.reduction == 'sum':
            return final_error.sum()
        else:
            return final_error                     # [B, T, N]

class InverseTransformProjectionLoss_v2(nn.Module):
    """
    Signed residual -> IDWT projection -> time-domain amplitude score

    核心修正：
    1. 不再先对小波系数做 MSE / L1 / LogCosh 这类“去符号”误差
    2. 先保留带符号的系数残差：pred_coeff - target_coeff
    3. 再通过 IDWT 投影回时域
    4. 最后在时域上计算逐点幅值分数

    返回:
        reduction='none' 时，返回 [B, T, N] 的逐点频域投影分数矩阵
    """

    def __init__(self, wavelet, level, seq_len, base_criterion=None, reduction='none'):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.seq_len = seq_len
        self.reduction = reduction

        # 这里的 base_criterion 不再用于“系数域误差”
        # 而是用于“IDWT 投影到时域之后的幅值评分”
        if base_criterion is None:
            self.base_criterion = LogCoshLoss(reduction='none')
        else:
            self.base_criterion = base_criterion
            if hasattr(self.base_criterion, "reduction"):
                self.base_criterion.reduction = 'none'

    def forward(self, pred_coeffs_list, target_coeffs_list):
        """
        pred_coeffs_list / target_coeffs_list:
            [cA, cD_L, cD_{L-1}, ..., cD_1]
            每个张量 shape 一般为 [B, N, T_i]

        return:
            reduction='none' -> [B, T, N]
        """
        assert len(pred_coeffs_list) == len(target_coeffs_list), \
            "pred_coeffs_list 与 target_coeffs_list 长度必须一致"

        # 1) 保留符号的小波系数残差
        signed_residual_coeffs = [pred_coeffs_list[0] - target_coeffs_list[0]]
        for i, (pred_c, target_c) in enumerate(zip(pred_coeffs_list[1:], target_coeffs_list[1:])):
            weight = 1.0 / (2 ** (i + 1))
            signed_residual_coeffs.append(weight * (pred_c - target_c))

        # 2) IDWT 投影回时域
        # ptwt.waverec 输出通常为 [B, N, T_recon]
        projected_residual = ptwt.waverec(signed_residual_coeffs, self.wavelet)

        # 3) 裁剪到原始长度
        # projected_residual = projected_residual[..., :self.seq_len]   # [B, N, T]
        if projected_residual.shape[-1] > self.seq_len:
            # 安全裁剪：去掉两端多余的部分（对称移除）
            excess = projected_residual.shape[-1] - self.seq_len
            start = excess // 2
            end = start + self.seq_len
            projected_residual = projected_residual[..., start:end]

        # 4) 转成 [B, T, N]，与主流程其余打分保持一致
        projected_residual = projected_residual.permute(0, 2, 1).contiguous()

        # 5) 在时域上做“幅值评分”
        #    目标是零残差，因此和 0 比较
        zero_target = torch.zeros_like(projected_residual)
        projected_score = self.base_criterion(projected_residual, zero_target)

        if self.reduction == 'mean':
            return projected_score.mean()
        elif self.reduction == 'sum':
            return projected_score.sum()
        else:
            return projected_score

# class InverseTransformProjectionLoss_v3(nn.Module):
#     """
#     【工业防爆极速版 v3】 自适应底噪门控频域雷达 (Wavelet Shrinkage + Nearest Upsampling)
#
#     核心物理逻辑质变：
#     1. 抛弃 IDWT 振铃伪影，通过最近邻插值强制对齐时域，确保异常边缘锐利如刀。
#     2. 全局动态底噪画像 (MAD)，联合 Train+Valid 过滤本底白噪音，0 误报。
#     3. [安全核心] 绝对物理底线 (Absolute Floor) 防御死线传感器导致的 MAD 坍缩。
#     4. [安全核心] 对数极值压缩 (Log-SNR) 驯服阶跃信号引发的分数爆炸，保护 Top-K 量纲平衡与 FP16 内存。
#     """
#
#     def __init__(self, wavelet, level, seq_len, base_criterion=None, reduction='none'):
#         super().__init__()
#         self.wavelet = wavelet
#         self.level = level
#         self.seq_len = seq_len
#         self.reduction = reduction
#
#         # 核心超参 1：触发异常放大的 MAD 倍数阈值 (3.0 属于鲁棒统计学极其安全的抗噪界限)
#         self.k_sigma = 3.0
#
#         # 🛡️ 核心超参 2：绝对物理底噪下限
#         # Scaled Space 中正常波动约在 1.0 级别，0.001 以下均视为死线或神经网络浮点误差。
#         self.absolute_mad_floor = 0.001
#
#         # 注册缓冲区保存各频带的全局底噪参数槽
#         self.global_mads = None
#         self.is_calibrated = False
#
#     def calibrate(self, model, loaders, device, max_batches=300):
#         """
#         全量稳健底噪校准接口（网格搜索生命周期内仅执行1次）：
#         提取每一层小波频带、每一个传感器的真实泛化底噪 (MAD)。
#         """
#         print(f"\n\t[Freq V3 Profiling] Calibrating Band-wise Noise Profile (Max {max_batches} batches)...")
#         model.eval()
#         all_residuals = []
#         batch_count = 0
#
#         with torch.no_grad():
#             for loader in loaders:
#                 if loader is None: continue
#                 for item in loader:
#                     if batch_count >= max_batches: break
#
#                     # 兼容不同的 DataLoader 格式
#                     batch_x = item[0] if isinstance(item, (list, tuple)) else item
#                     batch_x = batch_x.float().to(device)
#
#                     # 极速前向获取模型输出
#                     outputs = model(batch_x)
#                     recon_coeffs_list = outputs[1]
#
#                     target_coeffs_list = ptwt.wavedec(batch_x.permute(0, 2, 1), self.wavelet, level=self.level,
#                                                       mode='symmetric')
#
#                     # 转移至 CPU 内存防止显存 OOM
#                     batch_res = [torch.abs(p - t).cpu() for p, t in zip(recon_coeffs_list, target_coeffs_list)]
#                     all_residuals.append(batch_res)
#                     batch_count += 1
#
#                 if batch_count >= max_batches: break
#
#         if len(all_residuals) == 0:
#             print("\t[Warning] Calibration skipped due to empty loaders.")
#             return
#
#         num_bands = len(all_residuals[0])
#         num_channels = all_residuals[0][0].shape[1]
#
#         global_mads = []
#         for i in range(num_bands):
#             # 联合展平该频带所有 batch -> [Total_Samples * Length, Channels]
#             band_res = torch.cat([b[i].transpose(1, 2).reshape(-1, num_channels) for b in all_residuals], dim=0)
#
#             median = torch.median(band_res, dim=0)[0]
#             mad = torch.median(torch.abs(band_res - median), dim=0)[0]
#
#             # 🛡️ 联合防爆锁 1：双重下限截断
#             # 1. 相对下限：向当前频带的所有通道借用 5% 的平均方差
#             relative_floor = torch.mean(mad) * 0.05
#             # 2. 绝对下限：取相对下限与绝对物理下限 (0.001) 中的最大值，彻底封死零方差除零崩溃！
#             safe_min = max(relative_floor.item(), self.absolute_mad_floor)
#
#             mad_safe = torch.clamp(mad, min=safe_min)
#             global_mads.append(mad_safe.view(1, num_channels, 1))
#
#         self.global_mads = nn.ParameterList([nn.Parameter(m, requires_grad=False) for m in global_mads])
#         self.is_calibrated = True
#         print(
#             f"\t[Freq V3 Profiling] Done! Fortified noise profile locked for {num_bands} bands across {num_channels} channels.\n")
#
#     def forward(self, pred_coeffs_list, target_coeffs_list):
#         """
#         前向极速门控打分
#         返回: reduction='none' 时输出 [B, T, N] 的逐点高频异常暴击分数矩阵
#         """
#         assert len(pred_coeffs_list) == len(target_coeffs_list), "小波级数不一致"
#
#         B, N = pred_coeffs_list[0].shape[0], pred_coeffs_list[0].shape[1]
#         upsampled_band_scores = []
#
#         for i, (p, t) in enumerate(zip(pred_coeffs_list, target_coeffs_list)):
#             res = torch.abs(p - t)  # [B, N, L_i]
#
#             # 获取稳健底噪 MAD
#             if self.is_calibrated and self.global_mads is not None:
#                 mad = self.global_mads[i].to(res.device)
#             else:
#                 # 动态降级兜底方案（同享联合防爆护盾）
#                 res_flat = res.transpose(1, 2).reshape(B * res.shape[-1], N)
#                 median = torch.median(res_flat, dim=0)[0].view(1, N, 1)
#                 mad = torch.median(torch.abs(res_flat - median), dim=0)[0].view(1, N, 1)
#                 safe_min = max((torch.mean(mad) * 0.05).item(), self.absolute_mad_floor)
#                 mad = torch.clamp(mad, min=safe_min)
#
#             # 💡 核心打分：硬阈值门控暴击
#             raw_snr = F.relu(res - self.k_sigma * mad) / mad
#
#             # 🛡️ 防爆锁 2：对数极值压缩 (Log-SNR Compression)
#             # 完美驯服阶跃信号或剧烈攻击导致的突刺（比如 raw_snr = 10000 -> log1p(10000) = 9.21）
#             # 保证其可以与时域 LogCosh (通常为个位数) 在相同量纲下安全进行 Top-K 聚合，并保护 FP16 缓存引擎
#             snr_score = torch.log1p(raw_snr)
#
#             # 规避吉布斯振铃的最近邻绝对对齐 -> [B, N, seq_len]
#             score_up = F.interpolate(snr_score, size=self.seq_len, mode='nearest')
#             upsampled_band_scores.append(score_up)
#
#         # 跨频带无加权能量全局均值聚合 -> [B, N, Seq_Len]
#         # 由于各频带已经被 MAD 归一化为无量纲的 SNR 纯净倍数，无需再设定人工权重，直接物理均值合并
#         total_freq_score = torch.stack(upsampled_band_scores, dim=-1).mean(dim=-1)
#
#         # 轴翻转适配时域张量规范 -> [B, Seq_Len, N]
#         final_score = total_freq_score.permute(0, 2, 1).contiguous()
#
#         if self.reduction == 'mean':
#             return final_score.mean()
#         elif self.reduction == 'sum':
#             return final_score.sum()
#         else:
#             return final_score
#
#
# class InverseTransformProjectionLoss_v4(nn.Module):
#     """
#     【跨学科终极版 v4】 时频能量谱连续包络映射 (Continuous Time-Frequency Scalogram Projection)
#
#     1. 破解线性等价退化：在投影前，直接在小波系数域计算非线性特征（LogCosh），打破 IDWT 的线性陷阱，提取真正的独立频域畸变能量。
#     2. 软性底噪归一化：联合 Train+Valid 提取 MAD 进行软标准化，绝不用 ReLU 硬截断，完美保护并放大渐变软异常。
#     3. 连续物理包络对齐：抛弃 IDWT 振铃与 Nearest 马赛克。利用 1D 线性平滑插值 (mode='linear', align_corners=True)，
#        像起伏的山脉一样，完美重构物理波形的连续能量包络，与时域 Sobolev 动力学惩罚实现 100% 架构级协同。
#     """
#
#     def __init__(self, wavelet, level, seq_len, base_criterion=None, reduction='none'):
#         super().__init__()
#         self.wavelet = wavelet
#         self.level = level
#         self.seq_len = seq_len
#         self.reduction = reduction
#         self.eps = 1e-6
#
#         # 强制使用传入的时域基座非线性算子 (如 LogCoshLoss) 进行能量提取
#         if base_criterion is None:
#             from ts_benchmark.baselines.DTWAiF.utils.loss import LogCoshLoss
#             self.base_criterion = LogCoshLoss(reduction='none')
#         else:
#             self.base_criterion = base_criterion
#             if hasattr(self.base_criterion, "reduction"):
#                 self.base_criterion.reduction = 'none'
#
#         # 绝对物理底噪下限：防止极端平滑通道除零雪崩
#         self.absolute_mad_floor = 0.001
#         self.global_mads = None
#         self.is_calibrated = False
#
#     def calibrate(self, model, loaders, device, max_batches=300):
#         """全量稳健底噪校准：提取各频带【绝对残差】的固有本底白噪音基线。"""
#         print(f"\n\t[Freq V4 Profiling] Calibrating Band-wise Energy Envelope (Max {max_batches} batches)...")
#         model.eval()
#         all_residuals = []
#         batch_count = 0
#
#         with torch.no_grad():
#             for loader in loaders:
#                 if loader is None: continue
#                 for item in loader:
#                     if batch_count >= max_batches: break
#                     batch_x = item[0] if isinstance(item, (list, tuple)) else item
#                     batch_x = batch_x.float().to(device)
#
#                     outputs = model(batch_x)
#                     recon_coeffs_list = outputs[1]
#                     target_coeffs_list = ptwt.wavedec(batch_x.permute(0, 2, 1), self.wavelet, level=self.level,
#                                                       mode='symmetric')
#
#                     batch_res = [torch.abs(p - t).cpu() for p, t in zip(recon_coeffs_list, target_coeffs_list)]
#                     all_residuals.append(batch_res)
#                     batch_count += 1
#
#                 if batch_count >= max_batches: break
#
#         if len(all_residuals) == 0: return
#
#         num_bands = len(all_residuals[0])
#         num_channels = all_residuals[0][0].shape[1]
#
#         global_mads = []
#         for i in range(num_bands):
#             band_res = torch.cat([b[i].transpose(1, 2).reshape(-1, num_channels) for b in all_residuals], dim=0)
#             median = torch.median(band_res, dim=0)[0]
#             mad = torch.median(torch.abs(band_res - median), dim=0)[0]
#
#             # 双重下限托底防爆
#             safe_min = max((torch.mean(mad) * 0.05).item(), self.absolute_mad_floor)
#             mad_safe = torch.clamp(mad, min=safe_min)
#             global_mads.append(mad_safe.view(1, num_channels, 1))
#
#         self.global_mads = nn.ParameterList([nn.Parameter(m, requires_grad=False) for m in global_mads])
#         self.is_calibrated = True
#         print(f"\t[Freq V4 Profiling] Done! Continuous Envelope Profile locked for {num_bands} bands.\n")
#
#     def forward(self, pred_coeffs_list, target_coeffs_list):
#         assert len(pred_coeffs_list) == len(target_coeffs_list), "小波级数不一致"
#         B, N = pred_coeffs_list[0].shape[0], pred_coeffs_list[0].shape[1]
#         upsampled_band_scores = []
#
#         for i, (p, t) in enumerate(zip(pred_coeffs_list, target_coeffs_list)):
#             res = torch.abs(p - t)  # 提取绝对残差 [B, N, L_i]
#
#             # 获取稳健底噪 MAD
#             if self.is_calibrated and self.global_mads is not None:
#                 mad = self.global_mads[i].to(res.device)
#             else:
#                 res_flat = res.transpose(1, 2).reshape(B * res.shape[-1], N)
#                 median = torch.median(res_flat, dim=0)[0].view(1, N, 1)
#                 mad = torch.median(torch.abs(res_flat - median), dim=0)[0].view(1, N, 1)
#                 safe_min = max((torch.mean(mad) * 0.05).item(), self.absolute_mad_floor)
#                 mad = torch.clamp(mad, min=safe_min)
#
#             # 1. 软自适应信噪比归一化 (Soft Z-score Standardization)
#             # 无硬阈值拦截，用底噪柔和地归一化特征
#             z_score = res / mad
#
#             # 2. 纯正的频域非线性特征能量计算 (彻底打破线性悖论)
#             # 使用传入的 base_criterion (LogCosh) 计算能量。
#             # 这既能温柔保留 CalIt2 的微弱渐变波动，又能抑制极端突刺爆炸。
#             zero_target = torch.zeros_like(z_score)
#             band_energy = self.base_criterion(z_score, zero_target)
#
#             # 3. 🌟 连续物理包络对齐 (Continuous Envelope Upsampling)
#             # 抛弃 nearest 的断崖，采用 linear！
#             # 像物理波浪一样将低频能量点平滑映射到覆盖的时域区间上，物理连贯性极佳，杜绝引起 Sobolev 的暴走惩罚。
#             if band_energy.shape[-1] == self.seq_len:
#                 aligned_energy = band_energy
#             else:
#                 aligned_energy = F.interpolate(
#                     band_energy,
#                     size=self.seq_len,
#                     mode='linear',
#                     align_corners=True
#                 )
#
#             upsampled_band_scores.append(aligned_energy)
#
#         # 4. 跨频带无权重全局融合 (各频带已完成标准化，天然平权) -> [B, N, Seq_Len]
#         total_freq_score = torch.stack(upsampled_band_scores, dim=-1).mean(dim=-1)
#
#         # 5. 轴翻转适配时域张量规范 -> [B, Seq_Len, N]
#         final_score = total_freq_score.permute(0, 2, 1).contiguous()
#
#         if self.reduction == 'mean':
#             return final_score.mean()
#         elif self.reduction == 'sum':
#             return final_score.sum()
#         else:
#             return final_score

class InverseTransformProjectionLoss_v5(nn.Module):
    """
    【跨学科终极版 v5】 基于多分辨率分析 (MRA) 的子带独立能量提取与时域精确对齐方案

    1. 破解线性相消悖论：利用 MRA 将各频带残差独立通过 IDWT 投影回时域，先计算非线性能量 (LogCosh) 再叠加。
       数学原理：\sum NonLinear(IDWT(R_j)) \neq NonLinear(IDWT(\sum R_j))。赋予频域损失真正的独立非线性特征价值。
    2. 完美时域对齐 (零插值伪影)：坚决抛弃 nearest/linear 等破坏相位的插值算法。利用小波正交基底 (IDWT)
       完美重构各频带在时域的真实独立物理波形，时间相位 100% 对齐，绝对兼容 Sobolev 动力学导数。
    3. 守护物理量纲平衡：不使用 MAD 归一化，保证频域残差投影后的幅值与时域主特征
       处于绝对同一量纲 (O(0.01~0.5))，杜绝高频微弱白噪被放大数千倍导致 Top-K 分数核爆。
    4. 物理先验滤波：恢复对高频系数的指数衰减权重，充当理想的低通滤波器，完美过滤 Genesis/CalIt2 中的高频机械与环境本底白噪。
    """

    def __init__(self, wavelet, level, seq_len, base_criterion=None, reduction='none'):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.seq_len = seq_len
        self.reduction = reduction

        # 强制使用传入的时域基座非线性算子 (如 LogCoshLoss) 进行非线性能量提取
        if base_criterion is None:
            from ts_benchmark.baselines.DTWAiF.utils.loss import LogCoshLoss
            self.base_criterion = LogCoshLoss(reduction='none')
        else:
            self.base_criterion = base_criterion
            if hasattr(self.base_criterion, "reduction"):
                self.base_criterion.reduction = 'none'

    def forward(self, pred_coeffs_list, target_coeffs_list):
        assert len(pred_coeffs_list) == len(target_coeffs_list), "小波级数不一致"
        num_bands = len(pred_coeffs_list)

        time_domain_band_energies = []

        # MRA 多分辨率分析: 逐个频带独立重构并提取非线性能量
        for i in range(num_bands):
            # 1. 构建 MRA 隔离残差通道
            isolated_residual_coeffs = []
            for j in range(num_bands):
                if i == j:
                    # 仅保留当前频带 j 的残差
                    isolated_residual_coeffs.append(pred_coeffs_list[j] - target_coeffs_list[j])
                else:
                    # 其余所有频带强制填充 0 张量
                    isolated_residual_coeffs.append(torch.zeros_like(pred_coeffs_list[j]))

            # 2. IDWT 独立投影：将单一频带的纯净残差精确映射回时域 (无任何插值伪影，相位完美对齐原始序列)
            band_time_error = ptwt.waverec(isolated_residual_coeffs, self.wavelet)

            # 3. 裁剪序列边缘溢出 -> [B, N, seq_len] -> [B, seq_len, N]
            if band_time_error.shape[-1] > self.seq_len:
                band_time_error = band_time_error[..., :self.seq_len]
            band_time_error = band_time_error.permute(0, 2, 1).contiguous()

            # 4. 🌟 提取该独立子带在时域的非线性瞬时能量 (如 LogCosh)
            # 这一步彻底打破了 IDWT 的线性等价魔咒！且量纲与主时域损失 100% 同频对齐。
            zero_target = torch.zeros_like(band_time_error)
            band_energy = self.base_criterion(band_time_error, zero_target)

            # 5. 施加物理级低通衰减权重 (滤除 Genesis/CalIt2 中的高频自然抖动)
            weight = 1.0 if i == 0 else 1.0 / (2 ** i)

            time_domain_band_energies.append(band_energy * weight)

        # 6. 聚合 MRA 子带瞬时全频段能量 -> [B, Seq_Len, N]
        total_freq_energy = torch.stack(time_domain_band_energies, dim=-1).sum(dim=-1)

        if self.reduction == 'mean':
            return total_freq_energy.mean()
        elif self.reduction == 'sum':
            return total_freq_energy.sum()
        else:
            return total_freq_energy