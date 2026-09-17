
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import math
import ptwt

class LogCoshLoss(nn.Module):
    """
    Log-cosh error: quadratic near zero, linear in the tail, so a single large
    residual cannot dominate the gradient.
    """
    def __init__(self, reduction='mean'):
        super(LogCoshLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean'], "reduction must be 'none', 'sum' or 'mean'"
        self.reduction = reduction
        self.log2 = math.log(2.0)

    def forward(self, pred, target):
        abs_error = torch.abs(pred - target)
        loss = abs_error + F.softplus(-2.0 * abs_error) - self.log2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class TimeDomainSobolevLoss(nn.Module):
    """
    Amplitude error plus first- and second-difference penalties, so the
    reconstruction has to match the signal's dynamics and not only its level.
    """
    def __init__(self, alpha=0.5, beta=0.2, reduction='mean', base_criterion=None):
        super(TimeDomainSobolevLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean'], "reduction must be 'none', 'sum' or 'mean'"
        self.reduction = reduction
        self.alpha = alpha
        self.beta = beta
        if base_criterion is None:
            self.base_criterion = LogCoshLoss(reduction='none')
        else:
            self.base_criterion = base_criterion
            self.base_criterion.reduction = 'none'

    def forward(self, pred, target):
        loss_amp = self.base_criterion(pred, target)

        total_loss = loss_amp
        seq_len = pred.shape[1]
        if self.alpha > 0 and seq_len > 1:
            diff_1_pred = pred[:, 1:, :] - pred[:, :-1, :]
            diff_1_target = target[:, 1:, :] - target[:, :-1, :]
            loss_diff_1 = self.base_criterion(diff_1_pred, diff_1_target)

            loss_diff_1_padded = F.pad(loss_diff_1, (0, 0, 1, 0), mode='constant', value=0.0)
            total_loss = total_loss + self.alpha * loss_diff_1_padded

        if seq_len > 2 and self.beta > 0:
            diff_2_pred = diff_1_pred[:, 1:, :] - diff_1_pred[:, :-1, :]
            diff_2_target = diff_1_target[:, 1:, :] - diff_1_target[:, :-1, :]
            loss_diff_2 = self.base_criterion(diff_2_pred, diff_2_target)
            loss_diff_2_padded = F.pad(loss_diff_2, (0, 0, 2, 0), mode='constant', value=0.0)
            total_loss = total_loss + self.beta * loss_diff_2_padded

        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss

class WaveletDomainAuxiliaryLoss(nn.Module):
    """
    Band-weighted error between the wavelet coefficients of the reconstruction
    and those of the target.
    """
    def __init__(self, configs, reduction='mean',base_criterion=None):
        super(WaveletDomainAuxiliaryLoss, self).__init__()
        self.reduction = reduction
        if base_criterion is None:
            self.base_criterion = LogCoshLoss(reduction='none')
        else:
            self.base_criterion = base_criterion
            self.base_criterion.reduction = 'none'

    def forward(self, pred_coeffs_list, target_coeffs_list):
        weighted_losses = []
        loss_cA_elem = self.base_criterion(pred_coeffs_list[0], target_coeffs_list[0])
        weighted_losses.append(loss_cA_elem)

        for i, (p_cD, t_cD) in enumerate(zip(pred_coeffs_list[1:], target_coeffs_list[1:])):
            weight = 1.0 / (2 ** (i + 1))
            loss_cD_elem = weight * self.base_criterion(p_cD, t_cD)
            weighted_losses.append(loss_cD_elem)

        packed_weighted_loss = torch.cat(weighted_losses, dim=-1)

        if self.reduction == 'mean':
            return packed_weighted_loss.mean()
        elif self.reduction == 'sum':
            return packed_weighted_loss.sum()
        else:
            return packed_weighted_loss


class InverseTransformProjectionLoss_v2(nn.Module):
    """
    Per-timestamp, per-channel frequency-domain error obtained by projecting each
    band back to the time domain; this is the E_freq matrix used at scoring time.
    """
    def __init__(self, wavelet, level, seq_len, base_criterion=None, reduction='none'):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.seq_len = seq_len
        self.reduction = reduction

        if base_criterion is None:
            self.base_criterion = LogCoshLoss(reduction='none')
        else:
            self.base_criterion = base_criterion
            if hasattr(self.base_criterion, "reduction"):
                self.base_criterion.reduction = 'none'

    def forward(self, pred_coeffs_list, target_coeffs_list):
        assert len(pred_coeffs_list) == len(target_coeffs_list), \
            "pred_coeffs_list and target_coeffs_list must have the same length"

        signed_residual_coeffs = [pred_coeffs_list[0] - target_coeffs_list[0]]
        for i, (pred_c, target_c) in enumerate(zip(pred_coeffs_list[1:], target_coeffs_list[1:])):
            weight = 1.0 / (2 ** (i + 1))
            signed_residual_coeffs.append(weight * (pred_c - target_c))

        projected_residual = ptwt.waverec(signed_residual_coeffs, self.wavelet)

        if projected_residual.shape[-1] > self.seq_len:
            excess = projected_residual.shape[-1] - self.seq_len
            start = excess // 2
            end = start + self.seq_len
            projected_residual = projected_residual[..., start:end]

        projected_residual = projected_residual.permute(0, 2, 1).contiguous()

        zero_target = torch.zeros_like(projected_residual)
        projected_score = self.base_criterion(projected_residual, zero_target)

        if self.reduction == 'mean':
            return projected_score.mean()
        elif self.reduction == 'sum':
            return projected_score.sum()
        else:
            return projected_score
