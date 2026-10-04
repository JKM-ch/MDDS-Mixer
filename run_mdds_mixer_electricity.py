"""Run MDDS-Mixer on the Electricity benchmark with a dimension check."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HYPERPARAMETERS_BASE = {
    "task_name": "long_term_forecast",
    "is_training": 1,
    "model": "MDDS-Mixer",
    "root_path": "./data/electricity/",
    "data_path": "electricity.csv",
    "data": "custom",
    "features": "M",
    "target": "OT",
    "e_layers": 3,
    "patch_len": 16,
    "d_model": 8,
    "dropout": 0.1,
    "learning_rate": 0.0005,
    "batch_size": 128,
    "train_epochs": 50,
    "patience": 3,
    "itr": 1,
    "checkpoints": "./MDDS_Mixer_Electricity/",
}


def infer_channels() -> int:
    path = ROOT / "data" / "electricity" / "electricity.csv"
    with path.open("r", encoding="utf-8-sig") as handle:
        return len(handle.readline().rstrip("\r\n").split(",")) - 1


def build_command(seq_len: int, pred_len: int, channels: int, overrides=None) -> list[str]:
    model_id = f"Electricity_{seq_len}_{pred_len}"
    params = dict(HYPERPARAMETERS_BASE)
    params.update({"enc_in": channels, "dec_in": channels, "c_out": channels})
    if overrides:
        params.update(overrides)
    command = [sys.executable, "-u", "run.py"]
    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")
        else:
            command.extend([f"--{key}", str(value)])
    command.extend(["--model_id", model_id, "--seq_len", str(seq_len), "--pred_len", str(pred_len)])
    return command


def main() -> None:
    channels = infer_channels()
    print(f"Electricity data channels detected: {channels}")
    print("The manuscript reports 321 Electricity variables; use the exact 321-variable file to reproduce Table 3.")
    for pred_len in (96, 192, 336, 720):
        command = build_command(512, pred_len, channels, {"batch_size": 64, "use_amp": True})
        print("\n$ " + " ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
