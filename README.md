# MDDS-Mixer reproducibility package

This package contains the executable implementation and experiment scripts
for the PeerJ Computer Science article:

`MDDS-Mixer: Multi-scale dilated convolutional mixing for short- and long-term electricity load forecasting`

The code model is named `MDDS-Mixer` throughout this package. It combines
reversible instance normalization (RevIN), channel independence, multi-scale
causal dilated depthwise-separable convolution, and inter-patch/intra-patch
MLP mixing.

## Included datasets

Only the six datasets used in the manuscript are in scope:

- `ETTh1.csv` and `ETTh2.csv` (hourly ETT benchmarks)
- `ETTm1.csv` and `ETTm2.csv` (15-minute ETT benchmarks)
- `electricity.csv` (Electricity benchmark)
- `2016m15.csv` (short-term real-world grid-load data)

The local package includes all six files. For a public GitHub release, keep the ETT files in `data/ETT/` and remove the Electricity and M15
files in a private PeerJ supplement or keep them locally for
reproduction only. Do not publish either file on GitHub or Zenodo unless the
data license and provider permit redistribution. The `data/README.md` file
records the required provenance and access statement.

## Environment

The manuscript reports Python 3.8, PyTorch 2.0.0, CUDA 11.8, and an RTX
4090D. Install the pinned environment with:

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements_peerj.txt
```

## Long-term experiments

The long-term runner covers ETTh1, ETTh2, ETTm1, and ETTm2 with input length
512 and prediction lengths 96, 192, 336, and 720:

```bash
python run_mdds_mixer_long_peerj.py
```

The included local Electricity source file has 370 load columns. The manuscript reports 321 variables, so replace this file with the exact 321-variable release used for the paper before claiming numerical reproduction. The runner detects the local column count and will print this warning. The Electricity experiment uses the same prediction lengths:

```bash
python run_mdds_mixer_electricity.py
```

## Short-term experiment

The short-term runner uses M15 with input length 96 and prediction lengths 6,
12, 24, and 48:

```bash
python run_mdds_mixer_short_2016m15.py
```

It uses the chronological 70%/10%/20% train/validation/test split described
in the manuscript. Set `MODEL_FILTER=MDDS-Mixer` to run only the proposed
model; otherwise the configured baseline comparison is executed.

## Reproducibility records

For every reported table or figure, record the exact Git commit, command,
random seed, data version/checksum, Python/PyTorch/CUDA versions, and GPU.
The scripts write predictions, ground truth, metrics, and configuration
metadata to the output directories.

Before resubmission, replace the placeholder repository information with the
final GitHub URL, a version tag, a Zenodo DOI, the data licenses, and the
corresponding-author contact.
