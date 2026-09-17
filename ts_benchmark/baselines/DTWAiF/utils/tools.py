
import warnings

import numpy as np
import torch
import time
import copy
from scipy.stats import genpareto
from scipy.ndimage import gaussian_filter1d


def solve_pot(ref_scores, q=1e-3, tail_ratio=None, verbose=True, tag=""):
    """

    Peaks-over-threshold anomaly threshold.

    Fit a generalised Pareto distribution to the exceedances above a high empirical
    quantile and invert it at risk level q. Fitted on the reference pool, so q is
    the false-alarm rate on normal data.

    """
    from scipy.stats import genpareto

    _t = ('/' + tag) if tag else ''
    s = np.asarray(ref_scores, dtype=np.float64).ravel()
    _bad = ~np.isfinite(s)
    if _bad.any():
        warnings.warn(f"[POT{_t}] the reference scores contain {int(_bad.sum())} "
                      f"non-finite values (nan/inf); they were discarded.")
        s = s[~_bad]
    n = int(s.size)
    if n < 50:
        th = float(np.max(s)) if n else 0.0
        return th

    q = float(q)
    q_emp = float(np.clip(1.0 - q, 0.0, 1.0))

    _ss = np.sort(s)
    _k = int(np.floor(q * n))
    fallback = float(_ss[n - 1 - _k]) if _k < n else float(_ss[0])
    _fb_fpr = float((s > fallback).mean())

    if tail_ratio is None:
        tail_ratio = float(np.clip(5.0 * q, 0.02, 0.30))
    tail_ratio = float(np.clip(tail_ratio, 1.0 / n, 0.5))
    if q >= tail_ratio:
        return fallback

    u = float(np.quantile(s, 1.0 - tail_ratio))
    peaks = s[s > u] - u
    peaks = peaks[peaks > 1e-12]
    nt = int(peaks.size)

    if nt < 30:
        return fallback

    try:
        gamma, _loc, sigma = genpareto.fit(peaks, floc=0)
        p_target = 1.0 - (q * n / nt)
        if not (0.0 < p_target < 1.0):

            return fallback

        th = u + float(genpareto.ppf(p_target, gamma, loc=0.0, scale=sigma))
        if (not np.isfinite(th)) or th < u:
            return fallback

        return th

    except Exception as e:
        if verbose:
            warnings.warn(f"[POT{_t}] the GPD fit failed ({e}); using the empirical "
                          f"quantile = {fallback:.6g}")
        return fallback


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    """Learning-rate schedule selected by `args.lradj`."""
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
    """Track the best validation loss and keep a checkpoint of the best model."""
    def __init__(self, patience=7, delta=0):
        self.patience = patience
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
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss
