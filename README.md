---
title: Cricklytics
emoji: 🏏
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Cricket score prediction and analytics
---

# Cricklytics: Ball-by-Ball Cricket Analytics and Prediction System

Production-style cricket analytics platform built on ball-by-ball data for two competition modes:
- International T20 (World Cup style datasets)
- IPL

The system trains dual pipelines and serves real-time match intelligence through a Flask web app.

## Why This Project

Cricklytics is designed to answer practical in-match questions:
- What is the likely first-innings final total from current match state?
- What is the current second-innings win probability for the chasing side?
- How can ball-by-ball simulation and trend charts support tactical analysis?

It combines leakage-aware model training, calibrated probability modeling, and competition-specific artifacts.

## Core Capabilities

- First innings total projection (regression)
- Second innings win probability (classification + calibration)
- Second innings total projection (regression)
- Ball-by-ball match simulation engine
- Competition switching (International / IPL)
- Auto-generated analysis artifacts (leaderboards, summaries, feature importance, simulation CSVs, charts)

## System Architecture

Cricklytics uses a dual-model design with shared context:

1. First innings model
- Input: current score, wickets, overs, venue, teams, recent scoring context
- Output: projected first-innings final score

2. Second innings win model
- Input: chase state, target pressure, game context
- Output: calibrated win probability

3. Second innings score model
- Input: chase progression features
- Output: projected second-innings total

Training is grouped by match identity to reduce leakage risk and preserve realistic validation behavior.

## Tech Stack

- Python 3.12
- Flask
- scikit-learn
- XGBoost
- pandas, numpy
- matplotlib, seaborn

Dependencies are listed in [requirements.txt](requirements.txt).

## Docker Run

This repository includes a production-friendly Docker setup for serving the Flask app.

Files:
- [Dockerfile](Dockerfile)
- [requirements.docker.txt](requirements.docker.txt)
- [.dockerignore](.dockerignore)

Build image:

```powershell
docker build -t cricklytics-app .
```

Run container:

```powershell
docker run --rm -p 7860:7860 -e PORT=7860 cricklytics-app
```

Open:

```text
http://localhost:7860
```

### Hugging Face Spaces (Docker SDK)

1. Create a new Space and choose `Docker` as SDK.
2. Push this repository to the Space git remote.
3. Ensure `README.md` in the Space has Docker metadata:

```yaml
---
title: Cricket Score Predictor
emoji: 🏏
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 7860
---
```

## DVC + MLflow Operations

This project supports:
- MLflow tracking from training scripts (`train.py`, `train_ipl.py`)
- DVC stage-based reproducible training (`dvc.yaml`)

Core files:
- [dvc.yaml](dvc.yaml)
- [params.yaml](params.yaml)
- [scripts/dvc_train.py](scripts/dvc_train.py)

### One-time setup

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m dvc init
```

### Run with DVC (venv-safe)

International:

```powershell
.\.venv\Scripts\python.exe -m dvc repro train_international
```

IPL:

```powershell
.\.venv\Scripts\python.exe -m dvc repro train_ipl
```

Both:

```powershell
.\.venv\Scripts\python.exe -m dvc repro
```

### Check outputs

```powershell
Test-Path artifacts\metrics.json
Test-Path artifacts\ipl\metrics.json
.\.venv\Scripts\python.exe -m dvc metrics show
```

### If DVC lock error appears (Windows)

```powershell
$pidToCheck = <PID_FROM_ERROR>
Stop-Process -Id $pidToCheck -Force

if (Test-Path .dvc\tmp\rwlock) {
  Remove-Item .dvc\tmp\rwlock -Force
}
```

Then rerun the same `dvc repro` command using `.venv` python.

## Repository Layout

```text
.
|- app.py
|- train.py
|- train_ipl.py
|- train_full_pipeline.py
|- src/
|  |- components/
|  |- pipeline/
|  |- utils/
|- templates/
|- static/
|- artifacts/               # generated training outputs
|- logs/                    # runtime and training logs
|- t20_wc_2016_deliveries.csv
|- t20_wc_2021_deliveries.csv
|- t20_wc_2022_deliveries.csv
|- t20_wc_2024_deliveries.csv
|- IPL.csv                  # local-only large dataset (ignored by git)
```

## Data Inputs

International mode:
- Uses tournament files matching `t20_wc_*_deliveries.csv`
- `train.py` merges them into a consistent training source

IPL mode:
- Uses `IPL.csv` (or IPL delivery CSVs if available)
- `train_ipl.py` standardizes IPL schema before pipeline training

Note:
- `IPL.csv` is intentionally excluded from git because of GitHub size limits.

## Quick Start

1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

2. Install dependencies

```powershell
pip install -r requirements.txt
```

3. Start the web app

```powershell
python app.py
```

App runs with `use_reloader=False` and debug controlled by `FLASK_DEBUG=1`.

## Training Workflows

International training:

```powershell
python train.py
```

IPL training:

```powershell
python train_ipl.py
```

Full pipeline (direct source-combine flow):

```powershell
python train_full_pipeline.py
```

## Training Profiles and Runtime Control

Environment variables:

- `CRICKET_TRAIN_PROFILE`
  - `balanced` (recommended): faster than full, strong performance
  - `full`: heavier search, highest runtime
  - `fast`: minimal tuning, fastest

- `CRICKET_MAX_TUNING_ROWS`
  - Caps rows used for tuning on large datasets
  - Helpful for reducing runtime while keeping model quality

- `CRICKET_ARTIFACTS_DIR`
  - Controls artifact output root
  - IPL defaults to `artifacts/ipl`

Example (balanced + runtime cap):

```powershell
$env:CRICKET_TRAIN_PROFILE='balanced'
$env:CRICKET_MAX_TUNING_ROWS='20000'
python train_ipl.py
```

## Artifacts Produced

International:
- [artifacts/dual/dual_model_summary.json](artifacts/dual/dual_model_summary.json)
- model leaderboards
- feature importance CSVs
- simulation CSV and charts

IPL:
- [artifacts/ipl/dual/dual_model_summary.json](artifacts/ipl/dual/dual_model_summary.json)
- model leaderboards
- feature importance CSVs
- simulation CSV and charts

## Understanding Metrics

Main quality indicators:

- Win model:
  - ROC-AUC (higher is better)
  - Brier score (lower is better)

- Score models:
  - RMSE (lower is better)

General interpretation:
- IPL typically performs better due to larger, richer data volume
- International can be improved further by adding more non-World-Cup T20I data

## About Dual Charts

Charts under `artifacts/.../dual/charts` are generated automatically after training:
- score progression
- win probability trend
- run rate vs required rate
- wickets timeline
- momentum by over

They are useful for analysis and presentation, but not required for inference serving.

## Production Notes

- Models are trained with grouped validation to reduce leakage risk
- Probability outputs are calibration-aware
- Competition-specific artifacts prevent International/IPL output collisions
- Recommended default profile for practical retraining: `balanced`

## Roadmap

- Add broader International bilateral datasets
- Add time-aware validation splits for stronger realism
- Add model monitoring and drift checks
- Add CI pipeline for reproducible training checks

## License

Add your preferred license file (for example MIT) before public distribution.

---

If you use this project, please star the repository and reference Cricklytics in your derivative work.
