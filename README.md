# ClothSense

ClothSense is a local Fashion-MNIST research prototype for studying calibration,
conformal prediction, abstention, and uncertainty degradation under controlled
distribution shifts. Its photograph classifier is experimental and is not a
general-purpose clothing recognizer.

## Setup

Python 3.11 or newer is required. The project uses PyTorch/torchvision, NumPy,
pandas, scikit-learn/SciPy, Pillow, Matplotlib/Seaborn, PyYAML, FastAPI, and Uvicorn.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
```

## Project structure

```text
clothsense/                 research, preprocessing, inference, and API modules
configs/default.yaml        all experiment and upload settings
scripts/                    data, training, experiment, reporting, and demo commands
tests/                      focused Phase 1–6 tests
data/                       Fashion-MNIST, saved splits, and personal-photo input
artifacts/models/           checkpoints and uncertainty artifacts
artifacts/plots/            generated research figures
artifacts/processed_examples/ optional qualitative-demo outputs
results/                    SQLite database, CSVs, and summaries
```

## Reproduce the research pipeline

Prepare Fashion-MNIST and deterministic splits:

```powershell
python -m scripts.prepare_data
```

Train and evaluate models:

```powershell
python -m scripts.train --seed 17 --no-download
python -m scripts.train --all --no-download
python -m scripts.evaluate_clean --all --no-download
python -m scripts.fit_uncertainty --all --no-download
```

Run experiments:

```powershell
python -m scripts.run_experiments --smoke --no-download
python -m scripts.run_experiments --clean-only --no-download
python -m scripts.run_experiments --shifts-only --no-download
python -m scripts.run_experiments --all --no-download
```

Rebuild all five CSVs and aggregate statistics from SQLite without inference, then
regenerate figures and the compact summary:

```powershell
python -m scripts.export_results
python -m scripts.generate_plots
python -m scripts.summarize_results
python -m scripts.check_phase5
```

## Local image inference and API

Classify one JPG/JPEG/PNG directly. Add `--invert` when the source should be mapped
to bright foreground on dark background, and optionally choose a configured alpha.

```powershell
python -m scripts.classify_image data\personal_photos\example.jpg --invert --alpha 0.10
```

Run the direct/API consistency smoke check, then start the local API:

```powershell
python -m scripts.smoke_backend
python -m uvicorn clothsense.api:app --host 127.0.0.1 --port 8000
```

OpenAPI documentation is available at `http://127.0.0.1:8000/docs`. The API reads
saved checkpoints, uncertainty artifacts, CSVs, and chart files; it never retrains,
recalibrates, or retains ordinary uploads.

## Personal-photo qualitative workflow

Place approximately 20 personally sourced images in `data/personal_photos/` and,
if desired, create `labels.csv` from `labels.example.csv`. Then run:

```powershell
python -m scripts.process_demo_images
```

This creates compact qualitative output at `results/personal_photo_demo.json` and
processed 28×28 examples under `artifacts/processed_examples/`. Personal images are
ignored by Git and must not be treated as a statistical benchmark.

## Outputs

- Database: `results/clothsense.sqlite`
- Final CSVs: `results/*_metrics.csv`, `results/reliability_bins.csv`, and
  `results/risk_coverage.csv`
- Research figures: `artifacts/plots/01_*.png` through `10_*.png`
- Research summary: `results/research_summary.md`
- Report: `research_report.md`
