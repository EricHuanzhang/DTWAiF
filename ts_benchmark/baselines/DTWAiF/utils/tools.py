# Parts of this file are derived from the official CATCH implementation
# distributed with the TFB benchmark. The DTWAiF-specific modifications are
# noted in the docstrings below. All rights to the reused code remain with its
# original authors; see NOTICE for details.

import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import copy
from scipy.stats import genpareto
from scipy.ndimage import gaussian_filter1d

plt.switch_backend('agg')


def smooth_scores(scores, sigma=3):
    """
    【核心优化】对异常分数进行高斯平滑，消除毛刺
    """
    if sigma > 0:
        return gaussian_filter1d(scores, sigma=sigma)
    return scores


import numpy as np
from scipy.stats import genpareto


# def solve_pot(train_scores, q=1e-4):
#     """
#     【工业防弹版】POT (Peak Over Threshold) 动态极值阈值算法
#     修复了 loc 漂移、c=0 奇点崩溃、数值污染以及退化策略不自洽等落地漏洞
#     """
#     # 0. 数据安全清洗
#     train_scores = np.nan_to_num(np.array(train_scores).reshape(-1), nan=0.0)
#
#     # ==============================================================
#     # 确立全局绝对稳健的 Fallback (兜底) 标尺
#     # 坚决弃用易受极值污染的 mean 和 std，统一采用 Median + MAD 体系
#     # ==============================================================
#     def _get_robust_fallback():
#         median_val = np.median(train_scores)
#         mad_val = np.median(np.abs(train_scores - median_val))
#         # 1.4826 是将其缩放为等效正态分布标准差的常数
#         safe_mad = max(mad_val * 1.4826, 1e-5)
#         # 对于 RER-Gating 放大后的剧烈波动数据，5 倍安全标准差是极佳的后备红线
#         return median_val + 5 * safe_mad
#
#     robust_fallback_th = _get_robust_fallback()
#
#     # 1. 初始阈值 (取 98% 分位数，截取右长尾)
#     threshold = np.percentile(train_scores, 98)
#
#     # 2. 提取有效极值 (Active Peaks)
#     peaks = train_scores[train_scores > threshold] - threshold
#
#     # 🌟【致命修复 4】：严格过滤，去除等于或无限接近于 0 的微小浮点渣滓，防止矩阵奇异
#     peaks = peaks[peaks > 1e-6]
#
#     n = len(train_scores)
#     nt = len(peaks)
#
#     # 保护 1：极值样本太少，分布拟合失去统计意义，触发稳健兜底
#     if nt < 10:
#         print(f"  [POT Info] Too few active peaks ({nt}). Using Robust Fallback.")
#         return max(threshold, robust_fallback_th)
#
#     try:
#         # 3. 拟合 GPD 分布
#         # 🌟【致命修复 1】：必须强制锁定 loc=0 (floc=0)！
#         # 迫使极大似然估计只在 [0, +∞) 域内搜索 scale 和 c，严格契合您的数学公式前提
#         c, loc, scale = genpareto.fit(peaks, floc=0)
#
#         # 目标累积概率 (CDF 逆向推导: P(X < x | X > u))
#         p_target = 1.0 - (q * n / nt)
#         # 防止异常的风险系数 q 导致概率越界 [0, 1] 区间
#         p_target = max(0.0, min(p_target, 1.0 - 1e-10))
#
#         # 4. 计算极值红线
#         # 🌟【致命修复 2】：使用 SciPy 内置的 ppf (逆 CDF) 取代手写除法公式
#         # ppf 底层自带洛必达法则处理 c -> 0 的指数分布极限情况，彻底免疫除零崩溃
#         gpd_quantile = genpareto.ppf(p_target, c, loc=0, scale=scale)
#         th = threshold + gpd_quantile
#
#         # 保护 2：逻辑反转保护。POT 算出的红线绝不能比初始阈值还低，也不能是 NaN
#         if np.isnan(th) or np.isinf(th) or th < threshold:
#             print(f"  [POT Warning] EVT yielded invalid threshold ({th:.4f}). Using Fallback.")
#             return max(threshold, robust_fallback_th)
#
#         return th
#
#     except Exception as e:
#         # 🌟【致命修复 3】：哪怕拟合发生最底层的 C 语言库崩溃，也坚守鲁棒统计底线
#         print(f"  [POT Error] GPD fit failed ({e}). Using Robust Fallback.")
#         return max(threshold, robust_fallback_th)
import numpy as np
from scipy.stats import genpareto


def solve_pot(train_scores, q=1e-3):
    # 0. 数据安全清洗
    train_scores = np.nan_to_num(np.array(train_scores).reshape(-1), nan=0.0)

    def _get_robust_fallback():
        median_val = np.median(train_scores)
        mad_val = np.median(np.abs(train_scores - median_val))
        safe_mad = max(mad_val * 1.4826, 1e-5)
        return median_val + 5 * safe_mad

    robust_fallback_th = _get_robust_fallback()

    # 🌟【核心修复】：动态设定初始阈值，确保尾部容量至少是 q 的 5 倍
    # 如果 q=0.001，那么尾部至少取 0.5% (99.5 分位数)
    # 如果 q=0.02，那么尾部至少取 10% (90 分位数)
    tail_ratio = min(max(q * 5.0, 0.01), 0.15)
    percentile_val = 100.0 * (1.0 - tail_ratio)
    threshold = np.percentile(train_scores, percentile_val)

    # 2. 提取有效极值
    peaks = train_scores[train_scores > threshold] - threshold
    peaks = peaks[peaks > 1e-6]

    n = len(train_scores)
    nt = len(peaks)

    if nt < 10:
        return max(threshold, robust_fallback_th)

    try:
        c, loc, scale = genpareto.fit(peaks, floc=0)

        # 此时因为我们动态分配了 tail_ratio，nt/n 绝大概率远大于 q
        p_target = 1.0 - (q * n / nt)

        # 保护：如果因为数据高度重复导致 nt 依然极小，做安全保底
        if p_target <= 0:
            p_target = 0.5  # 如果 q 太大超出了实际尾部，保守取中位数

        p_target = max(0.0, min(p_target, 1.0 - 1e-10))
        gpd_quantile = genpareto.ppf(p_target, c, loc=0, scale=scale)
        th = threshold + gpd_quantile

        if np.isnan(th) or np.isinf(th) or th < threshold:
            return max(threshold, robust_fallback_th)

        return th

    except Exception as e:
        return max(threshold, robust_fallback_th)

# --- 以下保持原代码不变 ---

def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.learning_rate if epoch < 20 else args.learning_rate * (0.5 ** ((epoch // 20) // 1))}
    elif args.lradj == 'type5':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * (0.5 ** ((epoch // 10) // 1))}
    elif args.lradj == 'type6':
        lr_adjust = {20: args.learning_rate * 0.5 , 40: args.learning_rate * 0.01, 60:args.learning_rate * 0.01,8:args.learning_rate * 0.01,100:args.learning_rate * 0.01 }
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        self.check_point = copy.deepcopy(model.state_dict())
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

# def test_params_flop(model,x_shape):
#     """
#     If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
#     """
#     model_params = 0
#     for parameter in model.parameters():
#         model_params += parameter.numel()
#         print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
#     from ptflops import get_model_complexity_info
#     with torch.cuda.device(0):
#         macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
#         # print('Flops:' + flops)
#         # print('Params:' + params)
#         print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#         print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred

def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)