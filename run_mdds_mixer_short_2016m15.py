from __future__ import annotations

import copy
import json
import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from models import BP, BiGRU, CNN_GRU, GRU, LSTM, MDDS_Mixer, TCN, iTransformer
from utils.timefeatures import time_features

warnings.filterwarnings("ignore")


def set_seed(seed: int = 2021) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def smape(pred: np.ndarray, true: np.ndarray, eps: float = 1e-5) -> float:
    numerator = np.abs(pred - true)
    denominator = np.abs(pred) + np.abs(true) + eps
    return float(np.mean(2.0 * numerator / denominator) * 100.0)


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    return {
        "MAE": mae,
        "RMSE": rmse,
        "SMAPE": smape(pred, true),
        "R2": float(r2_score(true.reshape(-1), pred.reshape(-1))),
    }


class Dataset2016m15(Dataset):
    def __init__(
        self,
        root_path: str,
        data_path: str,
        flag: str,
        size: tuple[int, int, int],
        target: str = "OT",
        freq: str = "t",
    ) -> None:
        self.seq_len, self.label_len, self.pred_len = size
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.target = target
        self.freq = freq
        self.scaler = StandardScaler()
        self._read_data()

    def _read_data(self) -> None:
        csv_path = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(csv_path)

        date_col = next((c for c in df_raw.columns if c.lower() in {"date", "time", "timestamp"}), None)
        if date_col is not None:
            dt = pd.to_datetime(df_raw[date_col], errors="coerce").ffill().bfill()
            df_stamp_all = pd.DataFrame({"date": dt})
            df_raw = df_raw.drop(columns=[date_col])
        else:
            df_stamp_all = None

        df_raw = df_raw.ffill().bfill()

        if self.target in df_raw.columns:
            df_data = df_raw[[self.target]]
        else:
            df_data = df_raw.iloc[:, -1:]

        data = df_data.values.astype(np.float32)

        num_train = int(len(data) * 0.7)
        num_test = int(len(data) * 0.2)
        num_vali = len(data) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(data)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        train_data = data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if df_stamp_all is not None:
            df_stamp = df_stamp_all.iloc[border1:border2]
            stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=self.freq).transpose(1, 0)
        else:
            stamp = np.zeros((border2 - border1, 5), dtype=np.float32)
        self.data_stamp = stamp.astype(np.float32)

    def __len__(self) -> int:
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32),
            torch.tensor(seq_x_mark, dtype=torch.float32),
            torch.tensor(seq_y_mark, dtype=torch.float32),
        )


@dataclass
class Configs:
    model_name: str
    seq_len: int
    pred_len: int
    enc_in: int = 1
    dec_in: int = 1
    c_out: int = 1
    label_len: int = 48
    task_name: str = "short_term_forecast"
    dropout: float = 0.1
    learning_rate: float = 0.0005
    batch_size: int = 128
    train_epochs: int = 50
    patience: int = 3
    e_layers: int = 3
    d_layers: int = 1
    d_model: int = 8
    d_ff: int = 2048
    n_heads: int = 8
    factor: int = 1
    activation: str = "gelu"
    embed: str = "timeF"
    freq: str = "t"
    patch_len: int = 16
    stride: int = 8
    d_patch: int = 8
    use_norm: int = 1
    moving_avg: int = 25
    top_k: int = 5
    num_kernels: int = 6
    down_sampling_layers: int = 0
    down_sampling_window: int = 1
    down_sampling_method: str | None = None
    channel_independence: int = 1
    decomp_method: str = "moving_avg"
    output_attention: bool = False
    num_class: int = 0
    device: torch.device | None = None
    ablation_mode: str = "full_model"


MODEL_REGISTRY = {
    "MDDS-Mixer": MDDS_Mixer,
    "TCN": TCN,
    "GRU": GRU,
    "LSTM": LSTM,
    "BP": BP,
    "CNN_GRU": CNN_GRU,
    "iTransformer": iTransformer,
    "BiGRU": BiGRU,
}


def resolve_data_path() -> str:
    candidates = [
        "data/ETT/2016m15.csv",
        "data/ETT/本科毕设数据集/2016m15.csv",
        "2016m15.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Cannot find 2016m15.csv in expected locations.")


def build_dataloaders(
    data_path: str,
    seq_len: int,
    label_len: int,
    pred_len: int,
    batch_size: int,
) -> tuple[Dataset2016m15, DataLoader, DataLoader, DataLoader]:
    common_kwargs = {
        "root_path": "./",
        "data_path": data_path,
        "size": (seq_len, label_len, pred_len),
        "target": "OT",
        "freq": "t",
    }
    train_dataset = Dataset2016m15(flag="train", **common_kwargs)
    val_dataset = Dataset2016m15(flag="val", **common_kwargs)
    test_dataset = Dataset2016m15(flag="test", **common_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
    return test_dataset, train_loader, val_loader, test_loader


def make_decoder_input(batch_y: torch.Tensor, label_len: int, pred_len: int, device: torch.device) -> torch.Tensor:
    dec_zeros = torch.zeros_like(batch_y[:, -pred_len:, :], dtype=torch.float32)
    dec_inp = torch.cat([batch_y[:, :label_len, :], dec_zeros], dim=1)
    return dec_inp.to(device)


def train_one_model(
    model_name: str,
    model_module,
    pred_len: int,
    data_path: str,
    output_dir: Path,
    base_configs: Configs,
) -> dict[str, float | int | str]:
    run_dir = output_dir / f"2016m15_{model_name}_{pred_len}"
    run_dir.mkdir(parents=True, exist_ok=True)

    configs = copy.deepcopy(base_configs)
    configs.model_name = model_name
    configs.pred_len = pred_len

    test_dataset, train_loader, val_loader, test_loader = build_dataloaders(
        data_path=data_path,
        seq_len=configs.seq_len,
        label_len=configs.label_len,
        pred_len=pred_len,
        batch_size=configs.batch_size,
    )

    model = model_module.Model(configs).to(configs.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    early_stop_count = 0
    checkpoint_path = run_dir / "checkpoint.pth"

    train_start_time = time.time()
    for epoch in range(configs.train_epochs):
        model.train()
        epoch_losses = []

        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            optimizer.zero_grad()
            batch_x = batch_x.to(configs.device)
            batch_y = batch_y.to(configs.device)
            batch_x_mark = batch_x_mark.to(configs.device)
            batch_y_mark = batch_y_mark.to(configs.device)

            dec_inp = make_decoder_input(batch_y, configs.label_len, configs.pred_len, configs.device)
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            pred = outputs[:, -configs.pred_len:, :]
            true = batch_y[:, -configs.pred_len:, :]

            loss = criterion(pred, true)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
                batch_x = batch_x.to(configs.device)
                batch_y = batch_y.to(configs.device)
                batch_x_mark = batch_x_mark.to(configs.device)
                batch_y_mark = batch_y_mark.to(configs.device)

                dec_inp = make_decoder_input(batch_y, configs.label_len, configs.pred_len, configs.device)
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs[:, -configs.pred_len:, :]
                true = batch_y[:, -configs.pred_len:, :]
                val_losses.append(criterion(pred, true).item())

        avg_train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        avg_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        print(
            f"[{model_name:11s}][pred={pred_len:>2}] "
            f"epoch {epoch + 1:>2}/{configs.train_epochs} "
            f"train={avg_train_loss:.6f} val={avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_count = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            early_stop_count += 1
            if early_stop_count >= configs.patience:
                break

    train_time = time.time() - train_start_time

    if not checkpoint_path.exists():
        raise RuntimeError(f"No checkpoint saved for {model_name} pred_len={pred_len}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=configs.device))
    model.eval()

    preds = []
    trues = []
    infer_time_total = 0.0
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.to(configs.device)
            batch_y = batch_y.to(configs.device)
            batch_x_mark = batch_x_mark.to(configs.device)
            batch_y_mark = batch_y_mark.to(configs.device)

            dec_inp = make_decoder_input(batch_y, configs.label_len, configs.pred_len, configs.device)

            if torch.cuda.is_available() and configs.device.type == "cuda":
                torch.cuda.synchronize()
            infer_start = time.time()
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if torch.cuda.is_available() and configs.device.type == "cuda":
                torch.cuda.synchronize()
            infer_time_total += time.time() - infer_start

            preds.append(outputs[:, -configs.pred_len:, :].detach().cpu().numpy())
            trues.append(batch_y[:, -configs.pred_len:, :].detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    pred_shape = preds.shape
    preds_inv = test_dataset.scaler.inverse_transform(preds.reshape(-1, pred_shape[-1])).reshape(pred_shape)
    trues_inv = test_dataset.scaler.inverse_transform(trues.reshape(-1, pred_shape[-1])).reshape(pred_shape)

    metrics = compute_metrics(preds_inv, trues_inv)
    result = {
        "Dataset": "2016m15",
        "Model": model_name,
        "PredLen": pred_len,
        **metrics,
        "TrainSeconds": float(train_time),
        "InferMilliseconds": float(infer_time_total * 1000.0),
        "RunDir": str(run_dir),
    }

    np.save(run_dir / "pred.npy", preds_inv)
    np.save(run_dir / "true.npy", trues_inv)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def main() -> None:
    set_seed(2021)

    data_path = resolve_data_path()
    output_dir = Path(os.getenv("OUTPUT_DIR", "mdds_mixer_short"))
    output_dir.mkdir(parents=True, exist_ok=True)

    model_filter = os.getenv("MODEL_FILTER", "").strip()
    pred_lens_env = os.getenv("PRED_LENS", "").strip()

    model_names = list(MODEL_REGISTRY.keys())
    if model_filter:
        keep = {name.strip() for name in model_filter.split(",") if name.strip()}
        model_names = [name for name in model_names if name in keep]

    pred_lens = [6, 12, 24, 48]
    if pred_lens_env:
        pred_lens = [int(item.strip()) for item in pred_lens_env.split(",") if item.strip()]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_configs = Configs(
        model_name="MDDS-Mixer",
        seq_len=96,
        pred_len=6,
        enc_in=1,
        dec_in=1,
        c_out=1,
        device=device,
    )

    train_epochs_env = os.getenv("TRAIN_EPOCHS", "").strip()
    if train_epochs_env:
        base_configs.train_epochs = int(train_epochs_env)

    with open(output_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "2016m15",
                "data_path": data_path,
                "seq_len": base_configs.seq_len,
                "label_len": base_configs.label_len,
                "pred_lens": pred_lens,
                "models": model_names,
                "hyperparameters": {
                    "task_name": base_configs.task_name,
                    "dropout": base_configs.dropout,
                    "learning_rate": base_configs.learning_rate,
                    "batch_size": base_configs.batch_size,
                    "train_epochs": base_configs.train_epochs,
                    "patience": base_configs.patience,
                    "e_layers": base_configs.e_layers,
                    "d_model": base_configs.d_model,
                    "patch_len": base_configs.patch_len,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 96)
    print("2016m15 short-term comparison")
    print(f"data_path={data_path}")
    print(f"output_dir={output_dir}")
    print(f"device={device}")
    print(f"models={model_names}")
    print(f"pred_lens={pred_lens}")
    print("=" * 96)

    all_results: list[dict[str, float | int | str]] = []
    failed_runs: list[dict[str, str | int]] = []

    for pred_len in pred_lens:
        for model_name in model_names:
            try:
                result = train_one_model(
                    model_name=model_name,
                    model_module=MODEL_REGISTRY[model_name],
                    pred_len=pred_len,
                    data_path=data_path,
                    output_dir=output_dir,
                    base_configs=base_configs,
                )
                all_results.append(result)
                print(
                    f"[done] {model_name:11s} pred={pred_len:>2} "
                    f"MAE={result['MAE']:.4f} RMSE={result['RMSE']:.4f} "
                    f"SMAPE={result['SMAPE']:.4f} R2={result['R2']:.4f}"
                )
            except Exception as exc:
                failed_runs.append({"Model": model_name, "PredLen": pred_len, "Error": str(exc)})
                print(f"[fail] {model_name:11s} pred={pred_len:>2} error={exc}")

    if all_results:
        df = pd.DataFrame(all_results).sort_values(["PredLen", "MAE", "RMSE"])
        df.to_csv(output_dir / "2016m15_short_term_summary.csv", index=False)

        metric_view = df[["Model", "PredLen", "MAE", "RMSE", "SMAPE", "R2", "TrainSeconds", "InferMilliseconds"]]
        metric_view.to_csv(output_dir / "2016m15_short_term_metrics.csv", index=False)
        print("\nSummary:")
        print(metric_view.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if failed_runs:
        fail_df = pd.DataFrame(failed_runs)
        fail_df.to_csv(output_dir / "failed_runs.csv", index=False)
        print("\nFailed runs:")
        print(fail_df.to_string(index=False))


if __name__ == "__main__":
    main()
