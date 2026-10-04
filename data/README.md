# Dataset provenance and access

The manuscript evaluates MDDS-Mixer on ETTh1, ETTh2, ETTm1, ETTm2,
Electricity, and M15.

The local package stores the ETT benchmark files under `data/ETT/`; the full local package also contains the Electricity and M15 files. Cite their original
source and retain the original license.

The Electricity benchmark must be stored as `data/electricity/electricity.csv`
with the schema expected by the TSLib custom loader. The included local source has 370 load columns, whereas the manuscript reports 321; verify and replace it with the exact manuscript version before reproducing reported numbers. Record its source URL,
version, checksum, and redistribution terms.

The short-term M15 file must be stored as `data/ETT/2016m15.csv` with a date
column and an `OT` target column. The manuscript describes M15 as restricted
real-world grid data. Do not upload it publicly without written permission.
If access is restricted, provide a data-request procedure and preprocessing
script in the repository and state the restriction in the manuscript.

For each dataset, document sampling frequency, time range, variables,
missing-value handling, chronological split, normalization, and any
aggregation step.
