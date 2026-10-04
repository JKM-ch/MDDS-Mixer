"""Minimal forecasting data loaders used by the MDDS-Mixer experiments."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features


class _ForecastDataset(Dataset):
    def __init__(self, args, root_path, flag, size, features, data_path, target, scale, timeenc, freq):
        self.seq_len, self.label_len, self.pred_len = size
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self._read_data()

    def _read_data(self):
        path = os.path.join(self.root_path, self.data_path)
        df = pd.read_csv(path)
        date_col = df.columns[0]
        dates = pd.to_datetime(df[date_col])
        if self.features in ("M", "MS"):
            values = df.iloc[:, 1:].astype(np.float32).values
        else:
            values = df[[self.target]].astype(np.float32).values

        n = len(values)
        if getattr(self, "ett_factor", None) is not None:
            unit = 12 * 30 * 24 * self.ett_factor
            borders1 = [0, unit - self.seq_len, unit + 4 * 30 * 24 * self.ett_factor - self.seq_len]
            borders2 = [unit, unit + 4 * 30 * 24 * self.ett_factor, unit + 8 * 30 * 24 * self.ett_factor]
        else:
            num_train = int(n * 0.7)
            num_test = int(n * 0.2)
            num_val = n - num_train - num_test
            borders1 = [0, num_train - self.seq_len, n - num_test - self.seq_len]
            borders2 = [num_train, num_train + num_val, n]
        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(values[:borders2[0]])
            values = self.scaler.transform(values)
        else:
            self.scaler = None

        left, right = borders1[self.set_type], borders2[self.set_type]
        self.data_x = values[left:right]
        self.data_y = values[left:right]
        stamp = time_features(dates.iloc[left:right].values, freq=self.freq).transpose(1, 0)
        self.data_stamp = stamp.astype(np.float32)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        return (
            self.data_x[s_begin:s_end].astype(np.float32),
            self.data_y[r_begin:r_end].astype(np.float32),
            self.data_stamp[s_begin:s_end].astype(np.float32),
            self.data_stamp[r_begin:r_end].astype(np.float32),
        )


class Dataset_ETT_hour(_ForecastDataset):
    ett_factor = 1


class Dataset_ETT_minute(_ForecastDataset):
    ett_factor = 4


class Dataset_Custom(_ForecastDataset):
    pass
