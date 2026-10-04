"""Reproduce the MDDS_Mixer long-term ETT experiments."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PRED_LENGTHS = (96, 192, 336, 720)
DATASETS = (
    ("ETTh1", "ETTh1.csv", "h"),
    ("ETTh2", "ETTh2.csv", "h"),
    ("ETTm1", "ETTm1.csv", "t"),
    ("ETTm2", "ETTm2.csv", "t"),
)


def build_command(dataset: str, data_path: str, freq: str, pred_len: int) -> list[str]:
    model_id = f"MDDS_Mixer_{dataset}_sl512_pl{pred_len}"
    return [
        sys.executable,
        "-u",
        "run.py",
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--model_id",
        model_id,
        "--model",
        "MDDS-Mixer",
        "--data",
        dataset,
        "--root_path",
        "./data/ETT/",
        "--data_path",
        data_path,
        "--features",
        "M",
        "--target",
        "OT",
        "--freq",
        freq,
        "--seq_len",
        "512",
        "--label_len",
        "48",
        "--pred_len",
        str(pred_len),
        "--enc_in",
        "7",
        "--dec_in",
        "7",
        "--c_out",
        "7",
        "--e_layers",
        "3",
        "--patch_len",
        "16",
        "--d_model",
        "8",
        "--dropout",
        "0.2",
        "--learning_rate",
        "0.0005",
        "--batch_size",
        "128",
        "--train_epochs",
        "50",
        "--patience",
        "3",
        "--num_workers",
        "0",
        "--itr",
        "1",
        "--des",
        "mdds_mixer_peerj",
        "--checkpoints",
        "./outputs/checkpoints/",
    ]


def main() -> None:
    for dataset, data_path, freq in DATASETS:
        for pred_len in PRED_LENGTHS:
            command = build_command(dataset, data_path, freq, pred_len)
            print("\n$ " + " ".join(command))
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
