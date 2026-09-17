from typing import Tuple, Any, Optional
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ts_benchmark.baselines.time_series_library.utils.timefeatures import (
    time_features,
)
from ts_benchmark.utils.data_processing import split_before


class SlidingWindowDataLoader:
    """
    SlidingWindDataLoader class.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        batch_size: int = 1,
        history_length: int = 10,
        prediction_length: int = 2,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.history_length = history_length
        self.prediction_length = prediction_length
        self.shuffle = shuffle
        self.current_index = 0

    def __len__(self) -> int:
        return len(self.dataset) - self.history_length - self.prediction_length + 1

    def __iter__(self) -> "SlidingWindowDataLoader":
        if self.shuffle:
            self._shuffle_dataset()
        self.current_index = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.current_index >= len(self):
            raise StopIteration

        batch_inputs = []
        batch_targets = []
        for _ in range(self.batch_size):
            window_data = self.dataset.iloc[
                self.current_index : self.current_index
                + self.history_length
                + self.prediction_length,
                :,
            ]
            if len(window_data) < self.history_length + self.prediction_length:
                raise StopIteration

            inputs = window_data.iloc[: self.history_length].values
            targets = window_data.iloc[
                self.history_length : self.history_length + self.prediction_length
            ].values

            batch_inputs.append(inputs)
            batch_targets.append(targets)
            self.current_index += 1

        batch_inputs = torch.tensor(batch_inputs, dtype=torch.float32)
        batch_targets = torch.tensor(batch_targets, dtype=torch.float32)

        return batch_inputs, batch_targets

    def _shuffle_dataset(self):
        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)


def train_val_split(train_data, ratio, seq_len):
    if ratio == 1:
        return train_data, None

    elif seq_len is not None:
        border = int((train_data.shape[0]) * ratio)
        train_data_value, valid_data_rest = split_before(train_data, border)
        train_data_rest, valid_data = split_before(train_data, border - seq_len)
        return train_data_value, valid_data
    else:
        border = int((train_data.shape[0]) * ratio)
        train_data_value, valid_data_rest = split_before(train_data, border)
        return train_data_value, valid_data_rest


def decompose_time(time: np.ndarray, freq: str) -> np.ndarray:
    df_stamp = pd.DataFrame(pd.to_datetime(time), columns=["date"])
    freq_scores = {"m": 0, "w": 1, "b": 2, "d": 2, "h": 3, "t": 4, "s": 5}
    max_score = max(freq_scores.values())
    df_stamp["month"] = df_stamp.date.dt.month
    if freq_scores.get(freq, max_score) >= 1:
        df_stamp["day"] = df_stamp.date.dt.day
    if freq_scores.get(freq, max_score) >= 2:
        df_stamp["weekday"] = df_stamp.date.dt.weekday
    if freq_scores.get(freq, max_score) >= 3:
        df_stamp["hour"] = df_stamp.date.dt.hour
    if freq_scores.get(freq, max_score) >= 4:
        df_stamp["minute"] = df_stamp.date.dt.minute
    if freq_scores.get(freq, max_score) >= 5:
        df_stamp["second"] = df_stamp.date.dt.second
    return df_stamp.drop(["date"], axis=1).values


def get_time_mark(time_stamp: np.ndarray, timeenc: int, freq: str) -> np.ndarray:
    if timeenc == 0:
        origin_size = time_stamp.shape
        data_stamp = decompose_time(time_stamp.flatten(), freq)
        data_stamp = data_stamp.reshape(origin_size + (-1,))
    elif timeenc == 1:
        origin_size = time_stamp.shape
        data_stamp = time_features(pd.to_datetime(time_stamp.flatten()), freq=freq)
        data_stamp = data_stamp.transpose(1, 0)
        data_stamp = data_stamp.reshape(origin_size + (-1,))
    else:
        raise ValueError("Unknown time encoding {}".format(timeenc))
    return data_stamp.astype(np.float32)


def forecasting_data_provider(data, config, timeenc, batch_size, shuffle, drop_last):
    dataset = DatasetForTransformer(
        dataset=data,
        history_len=config.seq_len,
        prediction_len=config.horizon,
        label_len=config.label_len,
        timeenc=timeenc,
        freq=config.freq,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=drop_last,
    )
    return dataset, data_loader


class DatasetForTransformer:
    def __init__(
        self,
        dataset: pd.DataFrame,
        history_len: int = 10,
        prediction_len: int = 2,
        label_len: int = 5,
        timeenc: int = 1,
        freq: str = "h",
    ):
        self.dataset = dataset
        self.history_length = history_len
        self.prediction_length = prediction_len
        self.label_length = label_len
        self.current_index = 0
        self.timeenc = timeenc
        self.freq = freq
        self.__read_data__()

    def __len__(self) -> int:
        return len(self.dataset) - self.history_length - self.prediction_length + 1

    def __read_data__(self):
        df_stamp = self.dataset.reset_index()
        df_stamp = df_stamp[["date"]].values.transpose(1, 0)
        data_stamp = get_time_mark(df_stamp, self.timeenc, self.freq)[0]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.history_length
        r_begin = s_end - self.label_length
        r_end = r_begin + self.label_length + self.prediction_length

        seq_x = self.dataset[s_begin:s_end]
        seq_y = self.dataset[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x = torch.tensor(seq_x.values, dtype=torch.float32)
        seq_y = torch.tensor(seq_y.values, dtype=torch.float32)
        seq_x_mark = torch.tensor(seq_x_mark, dtype=torch.float32)
        seq_y_mark = torch.tensor(seq_y_mark, dtype=torch.float32)
        return seq_x, seq_y, seq_x_mark, seq_y_mark


class SegLoader(object):
    """
    Strictly optimized SegLoader based on utils.py logic.
    Features:
    1. Vectorized Pre-computation (Efficiency).
    2. Strict utils.py alignment:
       - Uses 'max_lookahead' for spectral analysis (High Recall).
       - Uses 'series_mean' by default (Noise Robustness).
       - Uses theoretically derived entropy mapping (se / ln(L)) instead of arbitrary 5.0.
       - Implements Detrending (Matches periodogram).
    """
    def __init__(self, data, win_size, step, labels=None, mode="train", config=None):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.data = data
        self.labels = labels
        self.config = config

        # Optional multi-scale window augmentation. DTWAiF does not pass a config
        # to this loader, so every one of these falls back to its default.
        self.use_adaptive_window = getattr(self.config, 'use_adaptive_window', False) if self.config else False
        self.adaptive_mix_prob = getattr(self.config, 'adaptive_mix_prob', 0.5) if self.config else 0.5
        self.min_scale = getattr(self.config, 'min_scale', 0.5) if self.config else 0.5
        self.max_scale = getattr(self.config, 'max_scale', 2.0) if self.config else 2.0

        self.channel_interaction = getattr(self.config, 'channel_interaction', 'series_mean')

        self.max_lookahead = int(self.win_size * self.max_scale)

        self.scale_profile = None
        if self.use_adaptive_window and self.mode == 'train':
            print(f"\nPre-computing Adaptive Window Profile | Strategy: {self.channel_interaction} | Base Window: {self.max_lookahead} | Mix Prob: {self.adaptive_mix_prob}")
            self.scale_profile = self._precompute_scale_profile()

    def _precompute_scale_profile(self):
        """
        Efficiently pre-computes scale profile using Vectorized RFFT.
        Logic strictly mirrors utils.py _get_adaptive_physical_len.
        """
        if isinstance(self.data, pd.DataFrame):
            raw_values = torch.tensor(self.data.values, dtype=torch.float32)
        else:
            raw_values = torch.tensor(self.data, dtype=torch.float32)

        if self.channel_interaction == 'series_mean':
            raw_values = torch.mean(raw_values, dim=1, keepdim=True)

        total_len = len(raw_values)
        entropy_scores = torch.zeros(total_len)

        ref_win = self.max_lookahead

        chunk_size = 5000
        pad_size = ref_win

        padded_data = F.pad(raw_values.transpose(0, 1), (0, pad_size), mode='replicate').transpose(0, 1)

        with torch.no_grad():
            for i in range(0, total_len, chunk_size):
                end = min(i + chunk_size, total_len)
                current_batch_len = end - i

                batch_data = padded_data[i : end + ref_win]
                windows = batch_data.unfold(0, ref_win, 1)

                if windows.shape[0] > current_batch_len:
                    windows = windows[:current_batch_len]

                windows_mean = torch.mean(windows, dim=-1, keepdim=True)
                windows = windows - windows_mean

                fft_res = torch.fft.rfft(windows, dim=-1)

                psd = fft_res.real.pow(2) + fft_res.imag.pow(2)

                psd_sum = torch.sum(psd, dim=-1, keepdim=True) + 1e-12
                pmf = psd / psd_sum

                entropy = -torch.sum(pmf * torch.log(pmf + 1e-12), dim=-1)

                if self.channel_interaction == 'series_mean':
                    final_entropy = entropy.squeeze(-1)
                elif self.channel_interaction == 'entropy_max':
                    final_entropy, _ = torch.max(entropy, dim=-1)
                elif self.channel_interaction == 'entropy_min':
                    final_entropy, _ = torch.min(entropy, dim=-1)
                elif self.channel_interaction == 'entropy_mean':
                    final_entropy = torch.mean(entropy, dim=-1)
                else:
                    final_entropy = entropy.squeeze(-1)

                entropy_scores[i:end] = final_entropy

        fft_bins = ref_win // 2 + 1
        max_theoretical_entropy = np.log(fft_bins)

        se = entropy_scores

        norm_se = torch.clamp(se / max_theoretical_entropy, 0, 1)

        scales = self.max_scale - norm_se * (self.max_scale - self.min_scale)

        return scales

    def __len__(self):
        if self.use_adaptive_window and self.mode == 'train':
            margin = self.max_lookahead
            return (self.data.shape[0] - margin) // self.step + 1
        else:
            if self.mode == "train":
                return (self.data.shape[0] - self.win_size) // self.step + 1
            elif self.mode == "val":
                return (self.data.shape[0] - self.win_size) // self.step + 1
            elif self.mode == "test":
                return (self.data.shape[0] - self.win_size) // self.step + 1
            else:
                return (self.data.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        use_adaptive_now = False

        if self.use_adaptive_window and self.mode == 'train':
            if random.random() < self.adaptive_mix_prob:
                use_adaptive_now = True

        if use_adaptive_now:
            scale = self.scale_profile[index] if index < len(self.scale_profile) else 1.0

            phys_len = int(self.win_size * scale)
            phys_len = min(phys_len, self.max_lookahead)

            min_len = 32
            phys_len = max(min_len, phys_len)

            end_index = index + phys_len
            start_index = index

            if end_index > len(self.data):
                end_index = len(self.data)
                start_index = end_index - phys_len
                if start_index < 0:
                    start_index = 0

            if isinstance(self.data, pd.DataFrame):
                raw_seq = self.data.iloc[start_index: end_index].values
            else:
                raw_seq = self.data[start_index: end_index]

            tensor_seq = torch.tensor(raw_seq).float().permute(1, 0).unsqueeze(0)

            resized_seq = F.interpolate(
                tensor_seq,
                size=self.win_size,
                mode='linear',
                align_corners=True
            )

            seq_data = resized_seq.squeeze(0).permute(1, 0).numpy()

            return np.float32(seq_data), np.float32(seq_data)

        else:
            if self.mode == "train":
                data_slice = self.data[index:index + self.win_size]
                return np.float32(data_slice), np.float32(self.labels[0:self.win_size])
            elif (self.mode == 'val'):
                data_slice = self.data[index:index + self.win_size]
                return np.float32(data_slice), np.float32(self.labels[0:self.win_size])
            elif (self.mode == 'test'):
                data_slice = self.data[index:index + self.win_size]
                return np.float32(data_slice), np.float32(self.labels[index:index + self.win_size])
            else:
                start = index // self.step * self.win_size
                end = start + self.win_size
                data_slice = self.data[start:end]
                return np.float32(data_slice), np.float32(self.labels[start:end])


# Build a SegLoader over a series. step is forced to 1 for train/val and to
# win_size for the threshold pass, so the flattened test scores are strictly
# time-ordered.
def anomaly_detection_data_provider(data, batch_size, win_size=100, step=100, labels=None, mode='train', config=None):
    dataset = SegLoader(data, win_size, 1, labels, mode, config=config)

    shuffle = False
    if mode == 'train' or mode == 'val':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0,
                             drop_last=False)
    return data_loader
