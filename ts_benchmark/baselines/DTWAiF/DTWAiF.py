"""
DTWAiF: a Dual-Track Wavelet Transformer for time series anomaly detection.

The detector reconstructs a multivariate series in the wavelet domain and turns
the reconstruction error into a per-timestamp anomaly score. Two tracks are kept
separate throughout the encoder: a *context* track carrying the approximation
band cA, and a *content* track carrying the full set of bands. Attention queries
and keys are drawn from the context track only, while values come from the
content track, so channel routing is decided by slow structure rather than by
high-frequency noise.

Pipeline
    detect_fit      train the reconstructor on the training split
    detect_score    per-timestamp anomaly scores for the test split
    detect_label    scores plus binary labels at a POT or percentile threshold

Scoring, once the model has produced a per-channel error matrix E[T, C]:
    _build_E              temporal smoothing, frequency-error fusion, per-channel
                          causal drift removal
    _aggregate_channels   multi-scale top-k aggregation with tail calibration
    _postprocess_scores   log1p compression onto a readable range
    _normalize_scores     sliding-baseline correction on the one-dimensional score
    solve_pot             extreme-value threshold fitted on the reference pool

The model interface (detect_fit / detect_score / detect_label) follows the
convention of the TFB benchmark and its CATCH baseline.
"""
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from torch.optim import lr_scheduler
import ptwt

from ts_benchmark.baselines.DTWAiF.models.DTWAiF_model import (
    DTWAiFModel,
)
from ts_benchmark.baselines.utils import anomaly_detection_data_provider, train_val_split
from ts_benchmark.baselines.DTWAiF.utils.loss import WaveletDomainAuxiliaryLoss, TimeDomainSobolevLoss, InverseTransformProjectionLoss_v2
from ts_benchmark.baselines.DTWAiF.utils.tools import EarlyStopping, adjust_learning_rate, solve_pot


# Defaults for every hyper-parameter. `search_*` entries are grid axes
# consumed by the benchmark strategy, not by the model itself.
DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS = {
    "lr": 0.0001,
    "Mlr": 0.00001,
    "e_layers": 3,
    "n_heads": 2,
    "d_ff": 256,
    "d_model": 128,
    "head_dim": 64,
    "dropout": 0.1,
    "auxi_lambda": 0.5,
    "score_lambda": 0.5,
    "regular_lambda": 0.5,
    "temperature": 0.07,
    "dc_lambda": 0.005,
    "num_epochs": 3,
    "batch_size": 32,
    "patience": 5,
    "anomaly_ratio": [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2, 3, 5.0, 10.0, 15, 20, 25],
    "channel_min_unique": 3,
    "drop_clock_channels": True,
    "clock_max_period": 512,
    "seq_len": 128,
    "pct_start": 0.3,
    "lradj": "type1",

    "use_pot": True,
    "pot_q": 0.001,

    "sobolev_alpha": -1.0,
    "sobolev_beta": -1.0,

    "discrete_threshold":20,
    "SMOOTH_WINDOW":7,
    "score_len_divisor": 8,
}

class TransformerConfig:
    """Attribute view over DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS, overridden by kwargs."""
    def __init__(self, **kwargs):
        for key, value in DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self):
        return self.seq_len

    @property
    def learning_rate(self):
        return self.lr


class HeterogeneousScaler:
    """

    Per-channel scaler that picks its transform from the channel's own statistics.

    A multivariate anomaly-detection panel usually mixes continuous sensors,
    low-cardinality command flags and constant columns. Forcing one variance
    normalisation on all of them inflates the quiet channels and drowns the
    informative ones, so each channel is classified first and scaled accordingly:

        constant        passed through unchanged
        discrete        min-max, so the level structure is preserved
        continuous      robust (median / IQR), so outliers do not set the scale

    `discrete_threshold` is the number of distinct values below which a channel
    counts as discrete.

    """
    def __init__(self, discrete_threshold=20, constant_tol=1e-6):
        self.discrete_threshold = discrete_threshold
        self.constant_tol = constant_tol

        self.scalers = {}

        self.last_valid_mode = None
        self.current_mode = None

        self.is_discrete_mask = None

    def fit(self, data):
        """Classify every channel and fit its scaler."""
        _, num_cols = data.shape

        if self.last_valid_mode is None:
            self.last_valid_mode = ['unfitted'] * num_cols
            self.scalers = {i: None for i in range(num_cols)}

        self.current_mode = ['unfitted'] * num_cols
        self.is_discrete_mask = np.zeros(num_cols, dtype=bool)

        for i in range(num_cols):
            col_data = data[:, i]
            col_clean = col_data[~np.isnan(col_data)]

            if len(col_clean) == 0:
                self.current_mode[i] = self.last_valid_mode[i]
                continue

            if (np.max(col_clean) - np.min(col_clean)) <= self.constant_tol:
                self.current_mode[i] = self.last_valid_mode[i]
            else:
                rounded_data = np.round(col_clean, decimals=5)
                unique_vals = np.unique(rounded_data)

                if len(unique_vals) <= self.discrete_threshold:
                    scaler = MinMaxScaler()
                    scaler.fit(col_clean.reshape(-1, 1))
                    self.scalers[i] = scaler
                    self.current_mode[i] = 'discrete'
                    self.last_valid_mode[i] = 'discrete'

                else:
                    scaler = RobustScaler()
                    scaler.fit(col_clean.reshape(-1, 1))

                    std_val = np.std(col_clean)

                    if scaler.scale_[0] < std_val * 0.05:
                        scaler.scale_[0] = max(std_val, 1e-3)

                    self.scalers[i] = scaler
                    self.current_mode[i] = 'continuous'
                    self.last_valid_mode[i] = 'continuous'

            if self.current_mode[i] == 'discrete':
                self.is_discrete_mask[i] = True

        return self

    def transform(self, data):
        """Apply the per-channel scalers fitted by :meth:`fit`."""
        scaled_data = data.astype(float)
        _, num_cols = data.shape

        for i in range(num_cols):
            mode = self.current_mode[i]
            scaler = self.scalers[i]

            if mode in ['continuous', 'discrete'] and scaler is not None:
                scaled_col = scaler.transform(data[:, i].reshape(-1, 1)).flatten()
                scaled_data[:, i] = scaled_col
            else:
                scaled_data[:, i] = data[:, i]

        return scaled_data

    def inverse_transform(self, scaled_data):
        """Map scaled values back to the original physical range."""
        original_data = scaled_data.copy()
        _, num_cols = scaled_data.shape

        for i in range(num_cols):
            mode = self.current_mode[i]
            scaler = self.scalers[i]

            if mode in ['continuous', 'discrete'] and scaler is not None:
                original_col = scaler.inverse_transform(scaled_data[:, i].reshape(-1, 1)).flatten()
                original_data[:, i] = original_col
            else:
                original_data[:, i] = scaled_data[:, i]

        return original_data

class DTWAiF:
    """

    Dual-track wavelet Transformer anomaly detector.

    The class constants below are estimator constants, not tunable
    hyper-parameters; they play the same role as the 1.4826 in a MAD estimator.

        _SN_LEN_DIVISOR   default divisor of the score-axis baseline window,
                          W = N // divisor (overridden by `score_len_divisor`)
        _SN_GRID_PER_WIN  sampling density of the sliding-baseline grid
        _SN_MIN_WINDOW    lower bound on any baseline window
        _SN_BASE_Q        quantile used for the baseline. Anomalies contaminate the
                          score one-sidedly, so a median (breakdown point 50%) is
                          not robust enough; Q40 raises the breakdown point to 60%.
        _CH_TAIL_Q        tail quantile used to calibrate the scale ladder
        _CH_LEN_DIVISOR   divisor of the channel-axis causal window. Larger than the
                          score-axis one because a causal window can fall entirely
                          inside an anomalous stretch.
        _CH_GRID_PER_WIN  sampling density of the per-channel causal baseline

    """
    _SN_LEN_DIVISOR = 8
    _SN_GRID_PER_WIN = 50
    _SN_MIN_WINDOW = 64
    _SN_BASE_Q = 40.0
    _CH_TAIL_Q = 95.0
    _CH_LEN_DIVISOR = 4
    _CH_GRID_PER_WIN = 20

    def __init__(self, **kwargs):
        super(DTWAiF, self).__init__()
        self.config = TransformerConfig(**kwargs)
        self.scaler = HeterogeneousScaler(discrete_threshold=self.config.discrete_threshold)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = self.config.seq_len

        # Wavelet-domain auxiliary loss plus a time-domain Sobolev loss:
        # amplitude, velocity and acceleration are matched together.
        self.wavelet_criterion = WaveletDomainAuxiliaryLoss(self.config, reduction='mean')
        self.criterion = TimeDomainSobolevLoss(self.config.sobolev_alpha, self.config.sobolev_beta, reduction='mean')

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by model.

        :return: An empty dictionary indicating that model does not require
            additional hyperparameters.
        """
        return {}

    def __repr__(self) -> str:
        """Returns a string representation of the model name."""
        return self.model_name

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        """Derive the data-dependent config fields (frequency, channel count) from the training split."""
        try:
            freq = pd.infer_freq(train_data.index)
        except Exception as ignore:
            freq = 'S'
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num
        self.config.label_len = 48

    def detect_validate(self, valid_data_loader, criterion):
        """Average validation loss, used by EarlyStopping to select the checkpoint."""
        self.model.eval()
        total_loss = []

        with torch.no_grad():
            for i, (batch_x, target) in enumerate(valid_data_loader):
                batch_x = batch_x.float().to(self.device)
                out_final, _, dcloss, _, _ = self.model(batch_x)

                time_loss = criterion(out_final, batch_x)

                target_wave_coeffs = self._extract_target_wavelet_features(batch_x)
                recon_packed_coeffs = self._extract_target_wavelet_features(out_final)

                auxi_loss = self.wavelet_criterion(recon_packed_coeffs, target_wave_coeffs)

                loss = time_loss + self.config.dc_lambda * dcloss + self.config.auxi_lambda * auxi_loss
                total_loss.append(loss.item())

        self.model.train()
        return np.average(total_loss)

    def _extract_target_wavelet_features(self, batch_normalized_x):
        """Wavelet coefficients"""
        with torch.no_grad():
            x_permuted = batch_normalized_x.permute(0, 2, 1)
            real_coeffs_list = ptwt.wavedec(x_permuted, self.model.wavelet, level=self.model.level, mode='symmetric')

        return real_coeffs_list

    def detect_fit(self, train_data: pd.DataFrame, train_label: pd.DataFrame):
        """

        Train the model.

        :param train_data: Training data used to train the model.
        :param train_label: Label used for training.

        """
        self.detect_hyper_param_tune(train_data)
        setattr(self.config, "task_name", "anomaly_detection")
        self.config.c_in = train_data.shape[1]
        self.training = True

        config = self.config
        self.scaler.fit(train_data.values)

        self._build_channel_mask(train_data.values)

        self.model = DTWAiFModel(self.config).to(self.device)

        train_data_value, valid_data = train_val_split(train_data, 0.8, None)

        train_data_value = pd.DataFrame(
            self.scaler.transform(train_data_value.values),
            columns=train_data_value.columns,
            index=train_data_value.index,
        )

        valid_data = pd.DataFrame(
            self.scaler.transform(valid_data.values),
            columns=valid_data.columns,
            index=valid_data.index,
        )

        self.valid_data_loader = anomaly_detection_data_provider(
            valid_data,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            labels=train_label,
            mode="val",
        )

        self.valid_data_loader_ordered = anomaly_detection_data_provider(
            valid_data,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            labels=train_label,
            mode="test",
        )

        self.train_data_loader = anomaly_detection_data_provider(
            train_data_value,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            labels=train_label,
            mode="train",
        )

        self.ref_loader = anomaly_detection_data_provider(
            train_data_value,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            labels=train_label,
            mode="thre",
        )

        self.ref_loader_val = anomaly_detection_data_provider(
            valid_data,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            labels=train_label,
            mode="thre",
        )

        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"\nTotal trainable parameters: {total_params}")

        if self.config.sobolev_alpha < 0.0 or self.config.sobolev_beta < 0.0:
            print("Auto-setting Sobolev alpha/beta based on training data dynamics...")
            self.auto_set_sobolev(self.train_data_loader)
        else:
            print(f"Using user-defined Sobolev alpha={self.config.sobolev_alpha}, beta={self.config.sobolev_beta}")

        self.early_stopping = EarlyStopping(patience=self.config.patience)

        train_steps = len(self.train_data_loader)
        main_params = [param for name, param in self.model.named_parameters() if 'mask_generator' not in name]

        self.optimizer = torch.optim.Adam(main_params,
                                          lr=self.config.lr)

        self.optimizerM = torch.optim.Adam(self.model.mask_generator.parameters(), lr=self.config.Mlr)

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            steps_per_epoch=train_steps,
            pct_start=self.config.pct_start,
            epochs=self.config.num_epochs,
            max_lr=self.config.lr,
        )

        schedulerM = lr_scheduler.OneCycleLR(
            optimizer=self.optimizerM,
            steps_per_epoch=train_steps,
            pct_start=self.config.pct_start,
            epochs=self.config.num_epochs,
            max_lr=self.config.Mlr,
        )

        time_now = time.time()

        for epoch in range(self.config.num_epochs):
            iter_count = 0
            epoch_loss_tracker = []

            epoch_time = time.time()
            self.model.train()

            step = min(int(len(self.train_data_loader) / 10), 100)
            for i, (batch_x, target) in enumerate(self.train_data_loader):
                iter_count += 1

                batch_x = batch_x.float().to(self.device)

                out_final, _, dcloss, _, _ = self.model(batch_x)

                time_loss = self.criterion(out_final, batch_x)

                target_wave_coeffs_list = self._extract_target_wavelet_features(batch_x)
                recon_coeffs_list = self._extract_target_wavelet_features(out_final)

                auxi_loss = self.wavelet_criterion(recon_coeffs_list, target_wave_coeffs_list)

                total_loss = time_loss + self.config.dc_lambda * dcloss + self.config.auxi_lambda * auxi_loss

                epoch_loss_tracker.append(total_loss.item())

                total_loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                if (i + 1) % step == 0:
                    self.optimizerM.step()
                    self.optimizerM.zero_grad()

                if (i + 1) % 10 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | training time loss: {2:.7f} | training fre loss: {3:.7f} | training dc loss: {4:.7f}".format(
                            i + 1, epoch + 1, time_loss.item(), auxi_loss.item(), dcloss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                            (self.config.num_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            total_loss = np.average(epoch_loss_tracker)
            valid_loss = self.detect_validate(self.valid_data_loader, self.criterion)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, total_loss, valid_loss
                )
            )

            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.optimizer, scheduler, epoch + 1, self.config)
            adjust_learning_rate(self.optimizerM, schedulerM, epoch + 1, self.config, printout=False)

    def _freq_needed(self):
        """
        Whether any positive score_lambda appears in the grid, i.e. whether
        the frequency-domain error matrix has to be computed at all.
        """
        try:
            g = getattr(self.config, 'search_score_lambdas', None)
            if g is None:
                return False
            if not isinstance(g, (list, tuple)):
                g = [g]
            return any((v is not None) and float(v) > 0 for v in g)
        except (TypeError, ValueError):
            return False

    def _get_raw_matrices(self, data_loader, series_name: str = None,
                          return_label: bool = False, return_details: bool = False,
                          cache_tag: str = None):
        """

        Run the model over a loader and return the raw per-channel matrices.

        :return: (E_time[T, C], E_freq[T, C] or None, recon[T, C] or None, label[T] or None)

        The four matrices are cached per (series, mode, cache_tag). Every
        post-processing hyper-parameter acts on the matrices rather than on the
        model, so a grid search over the scoring stage reuses one forward pass.

        """
        self.model.eval()

        if not hasattr(self, 'freq_proj_loss') and self.model is not None:
            self.freq_proj_loss = InverseTransformProjectionLoss_v2(
                wavelet=self.model.wavelet,
                level=self.model.level,
                seq_len=self.seq_len,
                base_criterion=nn.MSELoss(reduction='none'),
                reduction='none'
            )

        mode = getattr(data_loader.dataset, 'mode', 'unknown')
        key = f"{series_name}_{mode}_{cache_tag}" if cache_tag else f"{series_name}_{mode}"
        if not hasattr(self, '_raw_cache'):
            self._raw_cache = {}
        if key in self._raw_cache:
            return self._raw_cache[key]

        time_list, freq_list, recon_list, label_list = [], [], [], []
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.float().to(self.device)
                out_final, _, _dcloss, _attn, _mask = self.model(batch_x)

                raw_error = batch_x - out_final
                clipped_error = torch.clamp(raw_error, min=-1000.0, max=1000.0)
                raw_time_mse = torch.pow(clipped_error, 2)

                raw_freq_error = None
                if self.config.score_lambda > 0 or self._freq_needed():
                    target_coeffs = self._extract_target_wavelet_features(batch_x)
                    recon_coeffs = self._extract_target_wavelet_features(out_final)
                    raw_freq_error = self.freq_proj_loss(recon_coeffs, target_coeffs)

                C = raw_time_mse.shape[-1]
                time_list.append(raw_time_mse.reshape(-1, C).cpu().numpy().astype(np.float32))
                if raw_freq_error is not None:
                    freq_list.append(raw_freq_error.reshape(-1, C).cpu().numpy().astype(np.float32))
                if return_details:
                    recon_list.append(out_final.reshape(-1, C).cpu().numpy().astype(np.float32))
                if return_label:
                    lab, _ = torch.max(batch_y, dim=-1)
                    label_list.append(lab.reshape(-1).cpu().numpy())

        Et = np.concatenate(time_list, axis=0)
        Ef = np.concatenate(freq_list, axis=0) if len(freq_list) else None
        Rc = np.concatenate(recon_list, axis=0) if len(recon_list) else None
        Lb = np.concatenate(label_list, axis=0) if len(label_list) else None
        self._raw_cache[key] = (Et, Ef, Rc, Lb)

        return self._raw_cache[key]

    @staticmethod
    def _moving_average_full(X: np.ndarray, w: int):
        """Centred moving average along the time axis, length-preserving."""
        if w is None or w <= 1:
            return np.asarray(X, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        T, C = X.shape
        cs = np.concatenate([np.zeros((1, C), dtype=np.float64),
                             np.cumsum(X, axis=0, dtype=np.float64)], axis=0)
        pad = int(w) // 2
        idx = np.arange(T)
        lo = np.maximum(idx - pad, 0)
        hi = np.minimum(idx + pad + 1, T)
        return (cs[hi] - cs[lo]) / (hi - lo)[:, None]

    def _remove_channel_drift(self, E: np.ndarray, tag: str = ""):
        """

        Subtract each channel's own causal baseline from E[t, c].

        Distribution shift raises the error level of a drifting channel even when
        nothing anomalous happens there, which lets that channel win the top-k
        selection on level alone. Removing a per-channel causal baseline first makes
        the selection pick the channel that is genuinely the most wrong. The window is
        causal (trailing), so no future information enters the score.

        """
        E = np.maximum(np.asarray(E, dtype=np.float64), 0.0)
        N, C = E.shape
        if N < 4 * self._SN_MIN_WINDOW or C < 2:
            return E

        W = int(np.clip(N // self._CH_LEN_DIVISOR,
                        self._SN_MIN_WINDOW,
                        max(self._SN_MIN_WINDOW, N // 2)))
        stride = max(1, W // self._CH_GRID_PER_WIN)
        grid = np.arange(0, N, stride)
        if grid[-1] != N - 1:
            grid = np.append(grid, N - 1)

        g_med = np.percentile(E, self._SN_BASE_Q, axis=0)
        med_g = np.empty((len(grid), C), dtype=np.float64)
        for i, t in enumerate(grid):
            lo = max(0, t - W)
            seg = E[lo:max(lo + 1, t)]
            if len(seg) < 20:
                med_g[i] = g_med
            else:
                kk = min(max(int(len(seg) * self._SN_BASE_Q / 100.0), 0), len(seg) - 1)
                med_g[i] = np.partition(seg, kk, axis=0)[kk]

        idx = np.arange(N, dtype=np.float64)
        gf = grid.astype(np.float64)
        m = np.empty((N, C), dtype=np.float64)
        for c in range(C):
            m[:, c] = np.interp(idx, gf, med_g[:, c])
        m_bar = np.median(m, axis=0)

        drift = m - m_bar
        Ec = np.maximum(E - drift, 0.0)

        amp = np.abs(drift).max(axis=0) / np.maximum(m_bar, 1e-12)
        order = np.argsort(-amp)[:3]
        top = " ".join(f"ch{int(c)}:{amp[c]:.1f}x" for c in order if amp[c] > 0.5)

        return Ec

    @staticmethod
    def _coerce_num(v, default, cast, name):
        """

        Coerce a config entry to a number, falling back to `default` with a warning.

        The grid-search framework assigns list elements straight onto the config, so a
        single null in the hyper-parameter JSON would otherwise surface much later as
        an unrelated TypeError.

        """
        if v is None or (isinstance(v, str) and not v.strip()):
            warnings.warn(f"[Config] {name}=None, falling back to {default}. "
                          f"Check whether the hyper-parameter JSON contains a null value.")
            return cast(default)
        try:
            return cast(v)
        except (TypeError, ValueError):
            warnings.warn(f"[Config] {name}={v!r} is not convertible to "
                          f"{cast.__name__}, falling back to {default}.")
            return cast(default)

    def _build_channel_mask(self, train_values):
        """
        Select the channels that take part in scoring.
        """
        C = int(np.asarray(train_values).shape[1])
        self._chan_C = C
        thr = self._coerce_num(getattr(self.config, 'channel_min_unique', 3), 3,
                               int, 'channel_min_unique')
        if thr <= 0:
            self._chan_keep = None

            return
        X = np.asarray(train_values)
        n = int(min(len(X), max(20000, 0.2 * len(X))))
        nu = np.array([len(np.unique(X[:n, c])) for c in range(C)])
        keep = np.where(nu >= thr)[0]

        if bool(getattr(self.config, 'drop_clock_channels', False)) and len(keep) > 1:
            clock = [c for c in keep if self._is_clock_channel(X[:n, c])]
            if clock and len(clock) < len(keep):
                keep = np.array([c for c in keep if c not in set(clock)])
            elif clock:
                warnings.warn(f"[ChanMask] the clock rule matched all {len(clock)} channels; "
                              f"dropping none, since no channel would be left otherwise.")
        if len(keep) == 0:
            keep = np.array([int(np.argmax(nu))])
            warnings.warn(f"[ChanMask] the rules discarded all {C} channels; "
                          f"keeping ch{keep[0]}, the one with the most distinct values.")
        if len(keep) == C:
            self._chan_keep = None

            return
        self._chan_keep = keep
        drop = [int(c) for c in range(C) if c not in set(keep.tolist())]

    def _is_clock_channel(self, x):
        """True if the training split repeats exactly with some period P <= clock_max_period."""
        x = np.asarray(x).ravel()
        n = x.size
        maxP = self._coerce_num(getattr(self.config, 'clock_max_period', 512), 512,
                               int, 'clock_max_period')
        if n < 8 or maxP < 2:
            return False
        if len(np.unique(x)) > max(1024, n // 8):
            return False
        maxP = int(min(maxP, n // 3))
        k = min(64, n // 4)
        for P in range(2, maxP + 1):
            if not np.array_equal(x[P:P + k], x[:k]):
                continue
            if np.array_equal(x[P:], x[:-P]):
                return True
        return False

    def _build_E(self, Et, Ef, tag: str = ""):
        """

        Per-channel error matrix ready for aggregation.

            E = AvgPool(E_time, w) + score_lambda * AvgPool(E_freq, w)

        followed by per-channel causal drift removal. `w` is SMOOTH_WINDOW.

        """
        w = self._coerce_num(getattr(self.config, 'SMOOTH_WINDOW', 1), 1, int, 'SMOOTH_WINDOW')
        lam = self._coerce_num(getattr(self.config, 'score_lambda', 0.0), 0.0, float, 'score_lambda')
        if w < 1:
            w = 1
        keep = getattr(self, '_chan_keep', None)
        if keep is not None and Et is not None and Et.shape[1] == getattr(self, '_chan_C', Et.shape[1]):
            Et = Et[:, keep]
            if Ef is not None:
                Ef = Ef[:, keep]
        # Smoothing and fusion happen on the full time axis, never inside a
        # batch: both need the whole series to be well defined.
        E = self._moving_average_full(Et, w)
        if Ef is not None and lam > 0:
            E = E + lam * self._moving_average_full(Ef, w)
        E = self._remove_channel_drift(E, tag=tag)
        return E

    def _postprocess_scores(self, Et, Ef, tag: str = "", s_scale_override: float = None,
                            Q_override=None, E_pre=None):
        """

        E[T, C] -> (s_cmp[T], z[T, C]).

        Aggregate across channels, then compress with log1p(s / s_scale). The scale is
        the IQR of the positive part of the aggregated score, which places the scores
        in the linear region of log1p; passing `s_scale_override` makes a second
        series share the first one's scale so the two are comparable.

        """
        E = self._build_E(Et, Ef, tag=tag) if E_pre is None else E_pre
        s, z, k_star = self._aggregate_channels(E, tag=tag, Q_override=Q_override)

        keep = getattr(self, '_chan_keep', None)
        if keep is not None and z is not None and z.shape[1] == len(keep) \
                and len(keep) != getattr(self, '_chan_C', z.shape[1]):
            z_full = np.zeros((z.shape[0], int(self._chan_C)), dtype=z.dtype)
            z_full[:, keep] = z
            z = z_full

        s_raw_pos = np.maximum(s, 0.0)
        # log1p is only meaningful once the argument is O(1); the IQR of the
        # positive part supplies that scale.
        s_scale_self = max(float(np.subtract(*np.percentile(s_raw_pos, [75, 25]))), 1e-12)
        self._last_s_scale = s_scale_self
        s_scale = s_scale_self if s_scale_override is None else float(s_scale_override)
        _arg = s_raw_pos / s_scale
        s = np.log1p(_arg)
        _q75, _q25 = np.percentile(s, [75, 25])

        self._last_k_star = k_star

        return s, z

    def _ref_cache_key(self):
        """Cache key for the reference-pool matrices: everything upstream of channel aggregation."""
        w = self._coerce_num(getattr(self.config, 'SMOOTH_WINDOW', 1), 1, int, 'SMOOTH_WINDOW')
        lam = self._coerce_num(getattr(self.config, 'score_lambda', 0.0), 0.0, float, 'score_lambda')
        return int(w), float(lam)

    def _get_ref_Q(self, series_name):
        """
        First pass over the reference pool: calibrate the scale ladder.
        """
        key = (series_name,) + self._ref_cache_key()
        if not hasattr(self, '_ref_Q_cache'):
            self._ref_Q_cache, self._ref_E_cache = {}, {}
        if key in self._ref_Q_cache and key not in self._ref_E_cache:
            del self._ref_Q_cache[key]
        if key in self._ref_Q_cache:
            return self._ref_Q_cache[key]

        _saved = self._save_last_state()
        try:
            t0 = time.time()
            Et, Ef, _, _ = self._get_raw_matrices(self.ref_loader, series_name, cache_tag='ref')
            E_list = [self._build_E(Et, Ef, tag='ref')]
            if getattr(self, 'ref_loader_val', None) is not None:
                Etv, Efv, _, _ = self._get_raw_matrices(self.ref_loader_val, series_name,
                                                        cache_tag='refval')
                E_list.append(self._build_E(Etv, Efv, tag='refval'))
            E_all = np.concatenate(E_list, axis=0) if len(E_list) > 1 else E_list[0]
            ks = self._auto_scales(E_all.shape[1])
            A = self._multiscale_A(E_all, ks)
            # Tail quantile per scale: the calibration constant Q_k.
            Q = np.percentile(A, self._CH_TAIL_Q, axis=0)
            dt = time.time() - t0
        finally:
            self._restore_last_state(_saved)

        self._ref_E_cache = {key: E_list}
        self._ref_Q_cache[key] = Q

        return Q

    def _get_ref_energy(self, series_name, Q_ref, s_scale, iqr_g, W_test=None):
        """

        Second pass: measure the reference pool with the test segment's ruler.

        Three constants come from different places, because they do different things:

            Q_k       makes the max over scales a legitimate multiple comparison; it
                      must be estimated on normal data, hence the reference pool.
            s_scale   decides whether log1p operates in its linear or its logarithmic
                      region, so it has to follow the series being scored.
            iqr_g     a unit constant; it changes neither ranking nor any quantile.

        In short: the ruler is defined by the test segment, and the reference pool only
        says where "normal" falls on that ruler. `W_test` extends the same principle to
        the baseline window, without which the two segments subtract different amounts
        and their readings are not comparable.

        """
        key = (series_name,) + self._ref_cache_key() + (float(s_scale), float(iqr_g),
                                                        int(W_test) if W_test else 0)
        if not hasattr(self, '_ref_energy_cache'):
            self._ref_energy_cache = {}
        if key in self._ref_energy_cache:
            e_ref = self._ref_energy_cache[key]

            return e_ref

        _saved = self._save_last_state()
        try:
            parts = []
            for E, tg in zip(self._ref_E_cache[(series_name,) + self._ref_cache_key()],
                             ('ref', 'refval')):
                s_ref, _ = self._postprocess_scores(None, None, tag=tg, E_pre=E,
                                                   Q_override=Q_ref,
                                                   s_scale_override=s_scale)
                parts.append(np.maximum(
                    self._normalize_scores(s_ref, tag=tg, iqr_override=iqr_g,
                                           W_override=W_test), 0.0))
        finally:
            self._restore_last_state(_saved)

        e_ref = np.concatenate(parts) if len(parts) > 1 else parts[0]
        self._ref_energy_cache[key] = e_ref

        return e_ref

    def _save_last_state(self):
        """Snapshot the diagnostic attributes before scoring another segment."""
        return tuple(getattr(self, a, None) for a in
                     ('_last_k_star', '_last_score_baseline', '_last_score_scale',
                      '_last_drift_ratio', '_last_Q', '_last_s_scale', '_last_g_iqr'))

    def _restore_last_state(self, saved):
        """Restore the attributes saved by :meth:`_save_last_state`."""
        for a, v in zip(('_last_k_star', '_last_score_baseline', '_last_score_scale',
                         '_last_drift_ratio', '_last_Q', '_last_s_scale', '_last_g_iqr'), saved):
            setattr(self, a, v)

    def _auto_window(self, N: int):
        """

        Baseline window on the score axis: W = clip(N // divisor, 64, N // 2).

        `divisor <= 1` means no time-varying correction at all: the window covers the
        whole segment and the baseline degenerates to a global quantile. That setting
        is optimal on some datasets, so it has to remain reachable.

        """
        _div = self._coerce_num(getattr(self.config, 'score_len_divisor',
                                        self._SN_LEN_DIVISOR),
                                self._SN_LEN_DIVISOR, int, 'score_len_divisor')
        if _div < 1:
            _div = 1
        self._last_len_divisor = _div
        if _div <= 1:
            return N
        base = int(np.clip(N // _div,
                           self._SN_MIN_WINDOW,
                           max(self._SN_MIN_WINDOW, N // 2)))
        w = self._coerce_num(getattr(self.config, 'SMOOTH_WINDOW', 1), 1, int, 'SMOOTH_WINDOW')
        if w <= 1:
            return base
        if w * 2 > base:
            warnings.warn(f"[BaseWin] SMOOTH_WINDOW={w} approaches the baseline window "
                          f"W={base} (sw/W={w / base:.2f}). The smoothed score is then "
                          f"cancelled by its own sliding baseline; beyond sw/W of about "
                          f"0.6 the detector collapses. Suspect this first if the scores "
                          f"for this setting are unexpectedly poor.")
        return base

    def _grid_robust(self, X: np.ndarray, W: int):
        """

        Sliding median / IQR evaluated on a strided grid and interpolated back.

        The baseline is slow-varying by construction, so the interpolation loses
        nothing while the grid keeps the cost linear in N.

        :return: (med, iqr, ref), where ref is a per-channel typical local IQR that is
            immune to step-shaped drift.

        """
        X2 = X if X.ndim == 2 else X[:, None]
        N, C = X2.shape
        stride = max(1, W // self._SN_GRID_PER_WIN)
        min_periods = max(20, stride)
        grid = np.arange(0, N, stride)
        if grid[-1] != N - 1:
            grid = np.append(grid, N - 1)

        g_med = np.percentile(X2, self._SN_BASE_Q, axis=0)
        g_q75, g_q25 = np.percentile(X2, [75, 25], axis=0)
        g_iqr = np.maximum(g_q75 - g_q25, 1e-12)

        med_g = np.empty((len(grid), C), dtype=np.float64)
        iqr_g = np.empty((len(grid), C), dtype=np.float64)
        half = W // 2
        for i, g in enumerate(grid):
            lo, hi = max(0, g - half), min(N, g + half + 1)
            seg = X2[lo:hi]
            m = len(seg)
            if m < min_periods:
                med_g[i], iqr_g[i] = g_med, g_iqr
            else:
                kb = min(max(int(m * self._SN_BASE_Q / 100.0), 0), m - 1)
                kk = np.unique(np.array([m // 4, kb, (3 * m) // 4]))
                p = np.partition(seg, kk, axis=0)
                med_g[i] = p[kb]
                iqr_g[i] = p[(3 * m) // 4] - p[m // 4]

        idx = np.arange(N, dtype=np.float64)
        gf = grid.astype(np.float64)
        med = np.empty((N, C), dtype=np.float64)
        iqr = np.empty((N, C), dtype=np.float64)
        for c in range(C):
            med[:, c] = np.interp(idx, gf, med_g[:, c])
            iqr[:, c] = np.interp(idx, gf, iqr_g[:, c])
        ref = np.median(iqr_g, axis=0)
        if X.ndim == 1:
            return med[:, 0], iqr[:, 0], float(ref[0])
        return med, iqr, ref

    def _normalize_scores(self, u: np.ndarray, tag: str = "", iqr_override: float = None,
                          W_override: int = None):
        """

        Drift correction on the one-dimensional score: local location, global scale.

            s_energy(t) = [s_cmp(t) - med_W(s_cmp)(t)] / IQR_global(s_cmp)

        """
        u = np.asarray(u, dtype=np.float64)
        N = int(len(u))
        g_q75, g_q25 = np.percentile(u, [75, 25])
        g_iqr_self = float(max(g_q75 - g_q25, 1e-8))
        self._last_g_iqr = g_iqr_self
        # A constant denominator: it changes no ranking and no quantile, it only
        # puts two series on one readable scale.
        g_iqr = g_iqr_self if iqr_override is None else float(max(iqr_override, 1e-8))
        W_self = self._auto_window(N)
        self._last_base_W = W_self
        if W_override is None:
            W = W_self
        else:
            W = int(np.clip(W_override, self._SN_MIN_WINDOW,
                            max(self._SN_MIN_WINDOW, N // 2)))
            if int(W_override) != W:
                warnings.warn(f"[ScoreNorm{('/' + tag) if tag else ''}] W_override="
                      f"{int(W_override)} was clipped to W={W} (this segment has "
                      f"N={N}, upper bound N//2={N // 2}). The baseline window of this "
                      f"segment therefore differs from the test segment, so the "
                      f"threshold does not transfer exactly here.")
        if N < 4 * self._SN_MIN_WINDOW:
            warnings.warn(f"[ScoreNorm] N={N} is too short; falling back to global statistics.")
            return (u - float(np.median(u))) / g_iqr

        _div_now = self._coerce_num(getattr(self.config, 'score_len_divisor',
                                            self._SN_LEN_DIVISOR),
                                    self._SN_LEN_DIVISOR, int, 'score_len_divisor')
        if _div_now <= 1:
            # divisor <= 1: exact global constant. Running the grid with W = N is not
            # equivalent, because grid windows are truncated at the segment edges.
            _c = float(np.percentile(u, self._SN_BASE_Q))
            self._last_score_baseline = np.full(N, _c, dtype=np.float64)
            self._last_score_scale = np.full(N, g_iqr, dtype=np.float64)
            self._last_drift_ratio = 0.0

            return (u - _c) / g_iqr

        med, iqr, ref = self._grid_robust(u, W)
        scl = np.full(N, g_iqr, dtype=np.float64)

        D_abs = float(med.max() - med.min())
        D = float(D_abs / g_iqr)
        self._last_score_baseline, self._last_score_scale, self._last_drift_ratio = med, scl, D

        loc_med = float(np.median(iqr))
        rng_ratio = float(iqr.max() / max(loc_med, 1e-12))

        _hi = float(np.percentile(u, 90))
        _above = float((med > _hi).mean()) * 100.0

        return (u - med) / scl

    @staticmethod
    def _auto_scales(C: int):
        """Scale ladder for C channels, e.g. C=9 -> [1,2,3,4,6,8,9]."""
        C = max(1, int(C))
        ladder = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
        return sorted(set([k for k in ladder if k <= C] + [C]))

    @staticmethod
    def _multiscale_A(E, ks):
        """A_k(t): mean of the k largest channel errors at t, for every k in the ladder."""
        Es = np.sort(np.asarray(E, dtype=np.float64), axis=-1)[..., ::-1]
        cum = np.cumsum(Es, axis=-1)
        return np.take(cum, [k - 1 for k in ks], axis=-1) / np.array([float(k) for k in ks])

    def _aggregate_channels(self, E: np.ndarray, tag: str = "", Q_override=None):
        """

        E[T, C] -> (s_raw[T], z[T, C], k_star[T]).

            A_k(t)   = mean of the k largest channel errors at t
            s_raw(t) = max_k A_k(t) / Q_k,   Q_k = quantile(A_k, _CH_TAIL_Q)

        """
        E = np.maximum(np.asarray(E, dtype=np.float64), 0.0)
        N, C = E.shape
        ks = self._auto_scales(C)

        A = self._multiscale_A(E, ks)
        if Q_override is None:
            Q = np.percentile(A, self._CH_TAIL_Q, axis=0)
        else:
            Q = np.asarray(Q_override, dtype=np.float64).ravel()
            if Q.size != len(ks):
                warnings.warn(f"[ChannelAgg{('/' + tag) if tag else ''}] the supplied Q has "
                              f"length {Q.size}, which does not match the scale ladder "
                              f"({len(ks)}); calibrating on this series instead.")
                Q = np.percentile(A, self._CH_TAIL_Q, axis=0)
        Q = np.maximum(Q, max(1e-12, 1e-6 * float(A.max() if A.size else 1.0)))
        self._last_Q = Q.copy()
        R = A / Q

        s = R.max(axis=1)
        k_star = np.asarray(ks, dtype=np.float64)[R.argmax(axis=1)]

        z = E / Q[0]

        kq = np.percentile(k_star, [50, 90])
        share1 = float((k_star == ks[0]).mean()) * 100.0
        top_n = int(max(1, round(0.01 * N)))
        top_idx = np.argpartition(-s, min(top_n, N - 1))[:top_n]
        top_hist = [(int(k), float((k_star[top_idx] == k).mean()) * 100.0) for k in ks]
        top_str = " ".join(f"k{k}:{p:.0f}%" for k, p in top_hist if p >= 1.0)

        return s, z, k_star

    def detect_score(self, test: pd.DataFrame, test_labels: pd.DataFrame, series_name: str) -> np.ndarray:
        """

        Per-timestamp anomaly score for the test split.

        :param test: Test data.
        :param test_labels: Ground-truth labels, used for evaluation only.
        :param series_name: Name of the series, used as a cache key.
        :return: Anomaly scores aligned with the flattened test segment.

        """
        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")
        self.model.load_state_dict(self.early_stopping.check_point)

        self.model.to(self.device)
        self.model.eval()
        self.training = False

        self.thre_loader = anomaly_detection_data_provider(
            test,
            self.config.batch_size,
            self.config.seq_len,
            step=1,
            labels=test_labels,
            mode="thre"
        )

        _use_ref_calib = hasattr(self, 'ref_loader')
        Q_ref = self._get_ref_Q(series_name) if _use_ref_calib else None
        if not _use_ref_calib:
            warnings.warn("[RefCalib] no reference pool: Q_k is calibrated on the test "
                          "segment, so detect_score and detect_label no longer use the "
                          "same scoring convention.")
        _Et, _Ef, _, _ = self._get_raw_matrices(self.thre_loader, series_name,
                                                return_label=False, return_details=False)
        test_scores, _dim_scores = self._postprocess_scores(_Et, _Ef, tag='thre', Q_override=Q_ref)

        test_energy = self._normalize_scores(test_scores, tag="thre")

        return test_energy, test_energy

    def detect_label(self, test: pd.DataFrame, test_labels: pd.DataFrame, series_name: str) -> np.ndarray:
        """

        Anomaly scores plus binary labels for the test split.

        The threshold comes either from POT fitted on the reference pool (`use_pot`),
        or from percentiles of the reference pool joined with the test segment. Both
        anchor the threshold to data believed to be normal, so that the same nominal
        risk level means the same thing across datasets.

        :return: (predictions keyed by anomaly_ratio, anomaly scores)

        """
        raw_test_data = test.values.copy()

        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")
        self.model.load_state_dict(self.early_stopping.check_point)

        config = self.config
        self.model.to(self.device)
        self.model.eval()
        self.training = False

        self.thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            labels=test_labels,
            mode="thre",
        )

        _use_ref_calib = hasattr(self, 'ref_loader')
        Q_ref = self._get_ref_Q(series_name) if _use_ref_calib else None
        if not _use_ref_calib:
            warnings.warn("[RefCalib] no reference pool: Q_k is calibrated on the test "
                          "segment, which under-estimates the scores on datasets with a "
                          "high anomaly ratio and makes the POT threshold non-transferable.")

        _Et, _Ef, recon_scaled, thre_labels = self._get_raw_matrices(
            self.thre_loader, series_name, return_label=True, return_details=True)

        _ekey = (series_name, 'thre') + self._ref_cache_key()
        if getattr(self, '_E_cache_key', None) == _ekey:
            _E_test = self._E_cache

        else:
            _E_test = self._build_E(_Et, _Ef, tag='thre')
            self._E_cache_key, self._E_cache = _ekey, _E_test
        thre_scores, dim_scores = self._postprocess_scores(
            None, None, tag='thre', E_pre=_E_test, Q_override=Q_ref)

        test_energy = self._normalize_scores(thre_scores, tag="thre")
        test_energy = np.maximum(test_energy, 0.0)

        _test_s_scale = float(getattr(self, '_last_s_scale', 1.0))
        _test_iqr_g = float(getattr(self, '_last_g_iqr', 1.0))

        ref_energy = None
        if _use_ref_calib:
            _test_W = int(getattr(self, '_last_base_W', 0)) or None
            ref_energy = self._get_ref_energy(series_name, Q_ref, _test_s_scale, _test_iqr_g,
                                              W_test=_test_W)
            _rq = np.percentile(ref_energy, [50, 90, 99])
            _tq = np.percentile(test_energy, [50, 90, 99])
            _warn = ""
            if _rq[1] > _tq[1] * 1.02:
                _warn = ("  the reference pool's P90 exceeds the test segment's: clean data "
                         "should not score higher than data containing anomalies. Either the "
                         "two segments really are far apart (the model fails to extrapolate "
                         "to the test segment), or the calibration is not actually shared. "
                         "The POT threshold does not transfer here; use anomaly_ratio instead.")


        norm_dim_scores = np.maximum(np.asarray(dim_scores, dtype=np.float64), 0.0)

        if ref_energy is not None:
            # The percentile threshold is taken over reference pool + test segment, so
            # that anomaly_ratio means "a red line on the normal distribution" rather
            # than "a quota within the test segment".
            combined_energy = np.concatenate([ref_energy, test_energy], axis=0)
            _rf = len(ref_energy) / len(combined_energy)

        else:
            combined_energy = test_energy
            warnings.warn("[RatioPool] no reference pool; the threshold pool falls back "
                          "to the test segment itself.")

        recon_original = self.scaler.inverse_transform(recon_scaled)

        preds = {}

        if getattr(config, 'use_pot', False):
            print(f"Applying POT algorithm with q={config.pot_q}...")
            if ref_energy is None:
                warnings.warn("[POT] no reference pool found (was detect_fit skipped?); "
                              "falling back to the empirical %.3f%% quantile of the test "
                              "scores." % (100 * (1 - config.pot_q)))
                pot_threshold = float(np.quantile(test_energy, 1.0 - float(config.pot_q)))
            else:
                # Fit the GPD tail on the clean reference pool, then apply it to the test
                # segment. Fitting on the test tail instead would let the anomalies raise
                # the very threshold meant to catch them.
                pot_threshold = solve_pot(ref_energy, q=config.pot_q, tag='ref')
            plot_threshold = pot_threshold
            _hit = float((test_energy > pot_threshold).mean()) * 100.0

            preds[config.pot_q] = (test_energy > pot_threshold).astype(int)
            final_pred_labels = preds[config.pot_q]
        else:
            if not isinstance(self.config.anomaly_ratio, list):
                self.config.anomaly_ratio = [self.config.anomaly_ratio]
            plot_threshold = np.percentile(combined_energy, 100 - self.config.anomaly_ratio[0])
            for ratio in self.config.anomaly_ratio:
                threshold = np.percentile(combined_energy, 100 - ratio)
                preds[ratio] = (test_energy > threshold).astype(int)
            final_pred_labels = preds[self.config.anomaly_ratio[0]]

        return preds, test_energy

    def auto_set_sobolev(self, train_loader):
        """

        Calibrate the Sobolev weights from the training data's own dynamics.

        alpha and beta are set so that the first- and second-difference penalties enter
        the loss at the same order of magnitude as the amplitude term, which removes
        them from the tuning surface. A negative value in the config is the sentinel
        requesting this calibration.

        """
        self.model.eval()
        eps = 1e-12

        rms_x_list = []
        rms_d1_list = []
        rms_d2_list = []

        with torch.no_grad():
            for batch_x, _ in train_loader:
                x = batch_x.float().to(self.device)

                rms_x = torch.sqrt(torch.mean(x * x, dim=1) + eps)

                d1 = x[:, 1:, :] - x[:, :-1, :]
                rms_d1 = torch.sqrt(torch.mean(d1 * d1, dim=1) + eps)

                if x.shape[1] > 2:
                    d2 = d1[:, 1:, :] - d1[:, :-1, :]
                    rms_d2 = torch.sqrt(torch.mean(d2 * d2, dim=1) + eps)
                else:
                    rms_d2 = torch.zeros_like(rms_d1)

                rms_x_list.append(rms_x.reshape(-1).cpu())
                rms_d1_list.append(rms_d1.reshape(-1).cpu())
                rms_d2_list.append(rms_d2.reshape(-1).cpu())

        rms_x_all = torch.cat(rms_x_list, dim=0)
        rms_d1_all = torch.cat(rms_d1_list, dim=0)
        rms_d2_all = torch.cat(rms_d2_list, dim=0)

        Ex = float(torch.median(rms_x_all).item()) + 1e-12
        E1 = float(torch.median(rms_d1_all).item()) + 1e-12
        E2 = float(torch.median(rms_d2_all).item()) + 1e-12

        r1 = min(max(E1 / Ex, 0.0), 1.0)
        r2 = min(max(E2 / E1, 0.0), 1.0)

        alpha = (1.0 - r1)
        alpha = float(min(max(alpha, 0.05), 0.25))

        beta = (1.0 - r2)
        beta = float(min(max(beta, 0.00), 0.10))
        beta = float(min(beta, 0.5 * alpha))

        self.config.sobolev_alpha = alpha
        self.config.sobolev_beta = beta
        self.criterion = TimeDomainSobolevLoss(alpha, beta, reduction='mean')

        print(f"[auto_set_sobolev] Ex={Ex:.6g}, E1={E1:.6g}, E2={E2:.6g} -> alpha={alpha:.4f}, beta={beta:.4f}")

        return alpha, beta
