# ClothSense

ClothSense is a local research system for studying whether uncertainty estimates from an image classifier remain reliable when evaluation data differ from the training distribution. A compact convolutional neural network classifies the ten Fashion-MNIST categories; the same saved logits are then evaluated with raw softmax confidence, temperature scaling, standard and class-conditional split conformal prediction, and three abstention policies.

The central research question is:

> How do classification accuracy, calibration, conformal coverage, and selective risk change as Fashion-MNIST inputs undergo controlled distribution shifts?

The project includes the complete experimental pipeline, persisted results, a local inference API, and a React interface. It is not intended as a general-purpose clothing recognizer or an out-of-distribution detector.

## What the project implements

- Deterministic, stratified Fashion-MNIST train/validation/calibration splits with saved indices.
- Three independently initialized CNN runs using seeds 17, 42, and 73.
- Best-checkpoint selection by validation loss with early stopping.
- Calibration-only fitting of one scalar temperature and conformal thresholds per model.
- Standard and class-conditional split conformal prediction at alpha 0.05, 0.10, and 0.20.
- Confidence-threshold, singleton-set, and hybrid abstention policies.
- Fourteen controlled shifted-test scenarios plus the clean test scenario.
- Classification, calibration, conformal, selective-prediction, and runtime metrics.
- SQLite run metadata and five tidy CSV result exports.
- Student-t 95% confidence intervals across the three seeds.
- Ten static research figures generated from persisted CSV files.
- Deterministic JPG/JPEG/PNG preprocessing and artifact-backed inference.
- A minimal FastAPI service and four-page React/Vite research interface.

## System architecture

```text
Official Fashion-MNIST
        |
        +-- training (45,000) ----> CNN parameter fitting
        +-- validation (5,000) ---> early stopping / checkpoint selection
        +-- calibration (10,000) -> temperature and conformal thresholds
        +-- official test (10,000)
                    |
                    +-- clean and 14 configured shift scenarios
                    +-- one inference pass per model/scenario
                                   |
                                   +-- raw and calibrated probabilities
                                   +-- conformal prediction sets
                                   +-- abstention decisions
                                                |
                                                v
                                    SQLite -> CSV -> figures / API
                                                          |
                                                          v
                                                     React interface
```

The mathematical components are independent of persistence and presentation. FastAPI delegates preprocessing and prediction to `InferenceService`; React consumes the API and never trains, calibrates, or recomputes experiment metrics.

## Technology stack

| Layer | Implementation |
|---|---|
| Model and data | Python 3.11+, PyTorch, torchvision, NumPy |
| Metrics and statistics | scikit-learn, pandas, SciPy |
| Images and figures | Pillow, Matplotlib, Seaborn |
| Configuration | YAML loaded into Python dataclasses |
| Persistence | SQLite and CSV |
| Local API | FastAPI, Uvicorn, Pydantic schemas |
| Interface | React 19, React Router, Vite, CSS, Lucide icons |
| Tests | pytest and FastAPI TestClient |

## Experimental methodology

### Dataset separation

The official 60,000-image Fashion-MNIST training set is partitioned using split seed 2027. Saved indices are validated before reuse.

| Role | Samples | Permitted use |
|---|---:|---|
| Training | 45,000 | Gradient updates only |
| Validation | 5,000 | Early stopping and checkpoint selection |
| Calibration | 10,000 | Temperature and conformal fitting only |
| Clean test | 10,000 | Final clean evaluation and source for shifted scenarios |

All splits are stratified by class. The official test set is never included in the saved training-set partition. Normalization is computed from the 45,000-image training split only: mean `0.28602888`, standard deviation `0.35298434`. No training augmentation is configured.

### Model and training

The fixed model is:

```text
Conv2d(1, 32, 3x3, padding=1) -> ReLU -> MaxPool2d(2)
Conv2d(32, 64, 3x3, padding=1) -> ReLU -> MaxPool2d(2)
Flatten -> Linear(3136, 128) -> ReLU -> Dropout(0.50) -> Linear(128, 10)
```

Training uses cross-entropy loss, Adam with learning rate 0.001, batch size 128, at most 20 epochs, and patience 3. Validation loss is the sole checkpoint-selection metric. The selected upload model is seed 73 because it had the lowest validation loss (`0.2096`) among the three runs, not because of test performance.

### Uncertainty methods

- **Raw softmax:** uncalibrated ten-class probabilities and maximum probability.
- **Temperature scaling:** one strictly positive scalar fitted by calibration-set negative log-likelihood; dividing all logits by the scalar preserves the argmax.
- **Standard split conformal:** nonconformity score `1 - p(y|x)` and the finite-sample rank `ceil((n + 1)(1 - alpha))`.
- **Class-conditional conformal:** a separate threshold for each true calibration class; candidate class `k` uses class `k`'s threshold.
- **Abstention:** calibrated-confidence, singleton conformal-set, and hybrid singleton-plus-confidence rules.

The upload demonstration is configured at alpha `0.10` with the hybrid policy and confidence threshold `0.80`.

### Distribution shifts

All image corruptions operate in raw `[0,1]` pixel space, preserve the 28×28 shape and original label, and are normalized afterward.

| Family | Configured scenarios |
|---|---|
| Gaussian noise | standard deviation 0.10, 0.20, 0.30 |
| Rotation | 10°, 20°, 30° |
| Gaussian blur | sigma 0.5, 1.0, 1.5 |
| Brightness/contrast | brightness 0.80; brightness 0.60; contrast 1.50 |
| Class imbalance | classes 0, 1, 8, and 9 retained; other classes retained at 0.50 or 0.20 |

Together with the clean scenario, this produces 15 scenarios per model seed. The saved SQLite database contains 45 seed-specific scenarios, 1,575 completed runs, and 25,845 scalar metric records.

## Verified results

All values below were read from `results/aggregate_metrics.csv` and are means over seeds 17, 42, and 73. The CSV also contains the sample standard deviation and Student-t 95% confidence interval for every complete group.

### Clean test

| Quantity | Result |
|---|---:|
| Accuracy | 0.9173 (95% CI 0.9066–0.9279) |
| Raw ECE | 0.0159 |
| Temperature-scaled ECE | 0.0076 |
| Raw negative log-likelihood | 0.2426 |
| Temperature-scaled negative log-likelihood | 0.2329 |
| Standard conformal coverage, alpha 0.10 | 0.8971 |
| Standard worst-class coverage, alpha 0.10 | 0.6627 |
| Class-conditional worst-class coverage, alpha 0.10 | 0.8777 |
| Standard / class-conditional mean set size, alpha 0.10 | 0.9570 / 1.0123 |
| Hybrid-policy acceptance / selective risk | 0.8228 / 0.0230 |

Temperature scaling changed confidence quality without changing predicted classes. At alpha 0.10, class-conditional calibration improved clean worst-class coverage, with a modest increase in average prediction-set size.

### Strongest configured shifts

The conformal columns use standard conformal prediction at alpha 0.10.

| Scenario | Accuracy | Calibrated ECE | Coverage | Mean set size |
|---|---:|---:|---:|---:|
| Gaussian noise, std 0.30 | 0.4262 | 0.3702 | 0.3889 | 0.870 |
| Rotation, 30° | 0.3270 | 0.2891 | 0.2351 | 0.580 |
| Gaussian blur, sigma 1.5 | 0.7548 | 0.0795 | 0.6145 | 0.700 |
| Brightness, factor 0.60 | 0.8890 | 0.0106 | 0.8528 | 0.924 |
| Strong class imbalance | 0.9407 | 0.0093 | 0.9263 | 0.968 |

The class-imbalance accuracy increase reflects the changed class mixture and should not be interpreted as improved model quality. Strong noise and rotation show the central failure mode: accuracy, calibration, and conformal coverage can all deteriorate after distribution shift.

![Reliability diagram before and after temperature scaling](artifacts/plots/03_reliability_raw_vs_temperature.png)

![Conformal coverage under increasing shift severity](artifacts/plots/06_conformal_coverage_vs_shift_severity.png)

![Risk-coverage curves for abstention policies](artifacts/plots/09_risk_coverage.png)

These figures are generated from the saved CSVs with `python -m scripts.generate_plots`; experiment metrics are not recomputed by the plotting code.

## Setup

Run from PowerShell at the repository root.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The frontend additionally requires Node.js and npm:

```powershell
cd frontend
npm install
cd ..
```

## Reproduce the research pipeline

Prepare Fashion-MNIST, generate or reuse the deterministic split file, verify normalization, and create the sample grid:

```powershell
python -m scripts.prepare_data
```

Train all three configured seeds and fit the uncertainty artifacts:

```powershell
python -m scripts.train --all --no-download
python -m scripts.fit_uncertainty --all --no-download
```

Run the complete clean and shifted experiment matrix, then rebuild the exported results and figures:

```powershell
python -m scripts.run_experiments --all --no-download
python -m scripts.export_results
python -m scripts.generate_plots
python -m scripts.summarize_results
python -m scripts.check_phase5
```

The full pipeline trains three CNNs and evaluates the entire matrix. For a short integration check instead:

```powershell
python -m scripts.run_experiments --smoke --no-download
```

Useful narrower workflows are:

```powershell
python -m scripts.train --seed 17 --no-download
python -m scripts.evaluate_clean --all --no-download
python -m scripts.run_experiments --clean-only --no-download
python -m scripts.run_experiments --shifts-only --no-download
```

### Tests

The current test suite contains 42 focused tests covering split integrity, reproducibility, model training/checkpoint behavior, calibration and conformal invariants, abstention, shifts, metrics, storage/export/statistics, preprocessing, inference, and API behavior. The current checkout passes the full suite. Tests that exercise the production inference service require the selected saved checkpoint and uncertainty artifact.

```powershell
python -m pytest
```

Build-check the frontend with:

```powershell
cd frontend
npm run build
cd ..
```

## Run the local application

Start FastAPI from the repository root:

```powershell
.venv\Scripts\python.exe -m uvicorn clothsense.api:app --host 127.0.0.1 --port 8000
```

In a second PowerShell window, start Vite:

```powershell
cd frontend
npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api` and `/charts` to FastAPI. The interface provides Overview, Classify, Experiments, and Gallery pages.

The API exposes:

| Endpoint | Purpose |
|---|---|
| `GET /api/classes` | Class names and configured client metadata |
| `POST /api/classify` | In-memory classification of exactly one JPG/JPEG/PNG |
| `GET /api/results/summary` | Compact summary from saved aggregate results |
| `GET /api/results/charts` | Available generated research charts |

OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

### Classify one local image without the frontend

```powershell
python -m scripts.classify_image data\personal_photos\example.jpg --alpha 0.10 --invert
```

Upload preprocessing applies EXIF orientation, RGB conversion, centered black padding to a square, grayscale conversion, bilinear resize to 28×28, optional inversion, and the training normalization. Uploads are limited to 5,000,000 bytes, processed in memory, and not retained.

## Repository structure

```text
clothsense/                 Core data, model, uncertainty, shift, metric, storage,
                            experiment, reporting, inference, and API modules
configs/default.yaml        Seeds, paths, model/training values, shifts, uncertainty,
                            normalization, device, and upload settings
scripts/                    Data preparation, training, evaluation, experiment,
                            export, plotting, summary, smoke, and inference commands
tests/                      Focused unit and integration tests
frontend/                   React/Vite interface and API client
data/                       Fashion-MNIST files, saved splits, optional personal photos
artifacts/models/           Per-seed checkpoints, histories, logits, and uncertainty files
artifacts/plots/            Per-seed diagnostics and generated research figures
artifacts/processed_examples/ Optional qualitative processed-image outputs
results/                    SQLite database, five CSV exports, baselines, and summary
research_report.md          Concise interpretation of the completed experiments
```

Important result files are:

- `results/run_metrics.csv`: one row per run-level scalar metric.
- `results/aggregate_metrics.csv`: mean, sample standard deviation, and Student-t 95% confidence interval.
- `results/per_class_metrics.csv`: per-class classification and conformal metrics.
- `results/reliability_bins.csv`: fixed-bin confidence and empirical-accuracy data.
- `results/risk_coverage.csv`: acceptance and selective-risk points.
- `results/clothsense.sqlite`: models, scenarios, runs, statuses, and scalar metrics.

## Limitations

- Fashion-MNIST is the only dataset and contains centered 28×28 grayscale catalogue-style images rather than ordinary photographs.
- The study evaluates one compact CNN and only three random seeds.
- The shifts are controlled synthetic corruptions and deterministic class subsampling; they do not cover real-world distribution change broadly.
- Conformal coverage relies on calibration/evaluation exchangeability and is not guaranteed after distribution shift.
- Class-conditional calibration has fewer calibration examples per threshold.
- ECE depends on the configured ten confidence bins.
- The personal-photo workflow is prepared, but no personal-photo set is currently included; any such examples are qualitative rather than a benchmark.
- Abstention and prediction-set size communicate model uncertainty but do not constitute proven OOD detection.
- The API and interface are local research tools without authentication, deployment configuration, or production monitoring.

## Possible future work

Within a follow-on study, the current pipeline could be extended to more independent seeds, additional architectures, and externally collected natural-image evaluation data. Other defensible directions include shift-aware calibration protocols, explicit OOD-detection baselines evaluated separately from abstention, richer corruption models, and uncertainty comparisons across datasets. These capabilities are not implemented in the present repository.

For the complete experimental interpretation, see [`research_report.md`](research_report.md). The implementation plan and research safeguards are documented in [`clothsense-project-plan.md`](clothsense-project-plan.md).
