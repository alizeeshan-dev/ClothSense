# ClothSense: Project Specification and Implementation Plan

## 1. Project Definition

ClothSense is a local machine-learning research project that studies whether image-classification confidence remains trustworthy when input data changes. A small convolutional neural network will classify clothing into the ten Fashion-MNIST categories. The project will compare ordinary softmax confidence, temperature-scaled confidence, conformal prediction sets, and abstention under clean and shifted data.

The completed project will contain:

- One trained Fashion-MNIST classifier
- A reproducible corruption and distribution-shift pipeline
- Confidence calibration and conformal-prediction methods
- Abstention rules for uncertain predictions
- Statistical evaluation across repeated experiment seeds
- Experiment metadata stored in SQLite
- Detailed metrics and final CSV result files
- Research charts and a short results report
- A local React frontend connected to a minimal FastAPI backend
- An upload workflow for classifying clothing photographs

The project is a personal research prototype. It will run locally and will not be deployed, monetized, or designed for multiple users.

---

## 2. Primary Objective

Build and evaluate a clothing-image classifier that can communicate uncertainty instead of always returning one confident label, especially when its inputs differ from its training data.

---

## 3. Research Questions

1. How well calibrated is a CNN trained on Fashion-MNIST?
2. Does temperature scaling improve confidence calibration without changing classification accuracy?
3. Do split-conformal prediction sets achieve their requested coverage on clean test data?
4. How do calibration and conformal coverage deteriorate under image corruptions and class imbalance?
5. Does class-conditional conformal prediction improve worst-class coverage?
6. Can abstention reduce the error rate on accepted predictions?
7. How differently does the system behave on uploaded real-world clothing photographs compared with Fashion-MNIST images?

---

## 4. Fixed Scope

### Included

- Fashion-MNIST as the only core dataset
- One small CNN implemented in PyTorch
- Ten output classes
- Clean-data evaluation
- Five controlled distribution shifts
- Temperature scaling
- Standard split-conformal prediction
- Class-conditional split-conformal prediction
- Three simple abstention policies
- Repeated seeded experiments
- SQLite experiment storage
- CSV result exports
- Static research plots
- Local FastAPI backend
- Local React frontend
- Single-image JPG or PNG upload

### Excluded

- Deployment or cloud hosting
- User accounts, authentication, or authorization
- Payments or commercial features
- Multiple datasets
- Large pretrained vision models
- Object detection or image segmentation
- Clothing recommendations or shopping features
- Model retraining from the frontend
- Real-time camera or video processing
- Mobile application development
- Distributed training
- LLM, RAG, agent, reinforcement-learning, or genetic-algorithm features
- Production monitoring or production database design

---

## 5. Dataset and Class Labels

Use Fashion-MNIST. Each example is a centered 28 x 28 grayscale image belonging to one of these classes:

| ID | Class |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

Use deterministic, stratified splits from the official training set:

- Training split: 45,000 images
- Validation split: 5,000 images
- Calibration split: 10,000 images
- Clean test split: the official 10,000-image test set

The roles of the splits must remain separate:

- Training data fits CNN parameters.
- Validation data selects training settings and supports early stopping.
- Calibration data fits temperature scaling and conformal thresholds.
- Test data is used only for final evaluation and shifted-test generation.

No test labels may be used to tune the model, temperature, confidence thresholds, or conformal thresholds.

---

## 6. Model

Use one compact CNN so that the research focuses on uncertainty rather than model comparison.

Recommended architecture:

1. Convolution, 32 channels, 3 x 3 kernel, padding 1
2. ReLU
3. Max pooling, 2 x 2
4. Convolution, 64 channels, 3 x 3 kernel, padding 1
5. ReLU
6. Max pooling, 2 x 2
7. Flatten
8. Fully connected layer with 128 units
9. ReLU
10. Dropout
11. Fully connected output layer with 10 logits

Recommended training defaults:

- Loss: cross-entropy
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 128
- Maximum epochs: 20
- Early-stopping patience: 3 epochs
- Model selection metric: validation loss
- Fixed random seeds: 3 seeds for final experiments

Save:

- Best model weights for each seed
- Training and validation loss by epoch
- Training and validation accuracy by epoch
- Model configuration
- Dataset split indices

---

## 7. Prediction Methods

Every test image must be evaluated using the same trained CNN. The methods differ only in how the logits are interpreted.

### 7.1 Raw softmax

Apply softmax directly to the CNN logits.

Return:

- Highest-probability class
- Maximum softmax probability
- Complete ten-class probability vector

This is the uncalibrated baseline.

### 7.2 Temperature scaling

Fit one positive scalar temperature on the calibration split by minimizing negative log-likelihood. Divide logits by the learned temperature before applying softmax.

Temperature scaling must not change the class selected by argmax. It changes only the confidence values.

Return:

- Highest-probability class
- Maximum calibrated probability
- Complete calibrated probability vector
- Learned temperature

### 7.3 Standard split-conformal prediction

Use calibrated probabilities and the true-label nonconformity score:

`score(x, y) = 1 - calibrated_probability(y | x)`

For an error level `alpha`, calculate the finite-sample corrected conformal quantile from all calibration scores. Include class `k` in the prediction set when:

`1 - calibrated_probability(k | x) <= conformal_quantile`

Required alpha values:

- 0.05 for nominal 95% coverage
- 0.10 for nominal 90% coverage
- 0.20 for nominal 80% coverage

### 7.4 Class-conditional split-conformal prediction

Calculate a separate calibration-score quantile for each true class. When testing candidate class `k`, compare its score with the threshold calibrated from class `k`.

This method is intended to improve class-level coverage balance. Report both its benefit and the cost in prediction-set size.

---

## 8. Abstention Policies

An abstaining system may return `uncertain` instead of forcing one class prediction.

Implement these policies:

### Confidence threshold

Accept the top prediction only when its calibrated probability is at least a chosen threshold.

Evaluate thresholds:

- 0.50
- 0.60
- 0.70
- 0.80
- 0.90
- 0.95

### Singleton conformal set

Accept only when the conformal prediction set contains exactly one class. Abstain for empty or multi-class sets.

### Hybrid policy

Accept only when:

- The conformal set contains exactly one class, and
- The calibrated top-class probability meets a selected threshold

Use the experiment results to select one threshold for the upload demo. Store that value in configuration rather than hard-coding it in frontend code.

---

## 9. Distribution-Shift Scenarios

Generate shifted versions from the official test set. Never overwrite the clean test data.

### 9.1 Gaussian noise

Add zero-mean Gaussian pixel noise and clip pixels to the valid range.

Severity levels:

- Mild: standard deviation 0.10
- Medium: standard deviation 0.20
- Strong: standard deviation 0.30

### 9.2 Rotation

Rotate images while retaining the original 28 x 28 canvas.

Severity levels:

- Mild: 10 degrees
- Medium: 20 degrees
- Strong: 30 degrees

### 9.3 Gaussian blur

Apply Gaussian blur.

Severity levels:

- Mild: radius or sigma 0.5
- Medium: radius or sigma 1.0
- Strong: radius or sigma 1.5

### 9.4 Brightness and contrast

Create three deterministic combinations:

- Mild brightness reduction
- Strong brightness reduction
- Increased contrast

Store the exact numeric factors in the experiment configuration.

### 9.5 Class imbalance

Create imbalanced test sets by deterministic class-based subsampling. Keep every example from selected majority classes and retain smaller seeded fractions of the remaining classes.

Create at least two imbalance levels:

- Moderate imbalance
- Strong imbalance

Report macro metrics and per-class metrics so that majority classes do not hide weak minority-class performance.

### Shift-generation rules

- Preserve every retained image's original label.
- Use the same generated shifted datasets for every method within a seed.
- Save shift name, severity, parameters, seed, and sample count.
- Treat each shift and severity combination as a separate scenario.
- Do not retrain or recalibrate on shifted test data in the core experiment.

---

## 10. Uploaded Photograph Workflow

The frontend will allow a user to upload one JPG or PNG photograph of clothing. The system will attempt to map it to one of the ten Fashion-MNIST classes.

### 10.1 Validation

- Accept JPG, JPEG, and PNG only.
- Accept one image per request.
- Enforce a small configurable file-size limit.
- Reject unreadable or malformed files with a clear message.

### 10.2 Preprocessing

Create a deterministic preprocessing function shared by the API and tests:

1. Correct image orientation using metadata when available.
2. Convert the image to RGB for display.
3. Center-crop or pad it to a square.
4. Convert it to grayscale.
5. Resize it to 28 x 28 pixels.
6. Apply a configurable inversion option because Fashion-MNIST uses bright garments on a dark background.
7. Normalize it using the same values used during CNN training.

The frontend must display both the original upload and the exact processed 28 x 28 image sent to the model.

### 10.3 Inference response

Return:

- Predicted class
- Raw confidence
- Calibrated confidence
- Top three calibrated classes and probabilities
- Standard conformal set
- Class-conditional conformal set
- Selected alpha value
- Abstention decision
- Short explanation of the decision

Example:

```text
Predicted class: Coat
Raw confidence: 0.91
Calibrated confidence: 0.72
Conformal set: [Coat, Pullover, Shirt]
Decision: Uncertain
Reason: The conformal set contains multiple plausible classes.
```

### 10.4 Interpretation constraint

Fashion-MNIST contains centered grayscale catalogue-style images, not ordinary phone photographs. A real photograph may therefore be far outside the training distribution. The interface must describe the result as an experimental classification and must not claim that the model can reliably identify every kind of clothing.

Abstention or a large conformal set indicates uncertainty. It must not be presented as a mathematically proven out-of-distribution detector.

---

## 11. Evaluation Metrics

### 11.1 Classification

- Accuracy
- Macro F1 score
- Per-class precision
- Per-class recall
- Per-class F1 score
- Confusion matrix

### 11.2 Probabilistic quality

- Negative log-likelihood
- Brier score
- Expected calibration error using fixed confidence bins
- Maximum calibration error
- Mean confidence
- Accuracy-confidence gap
- Reliability-diagram bin values

The number and boundaries of calibration bins must be fixed in configuration and identical across compared methods.

### 11.3 Conformal prediction

- Empirical marginal coverage
- Coverage gap: empirical coverage minus target coverage
- Average prediction-set size
- Median prediction-set size
- Singleton-set rate
- Empty-set rate
- Multi-class-set rate
- Per-class coverage
- Worst-class coverage
- Per-class average set size

### 11.4 Abstention and selective prediction

- Acceptance rate
- Abstention rate
- Selective accuracy on accepted predictions
- Selective risk, equal to one minus selective accuracy
- Number of accepted samples
- Risk-coverage curve

### 11.5 Runtime

- Model-training time
- Calibration-fitting time
- Evaluation time per scenario
- Mean local inference time per image

Runtime is descriptive only and is not the main research outcome.

---

## 12. Statistical Experiment Design

Run the complete training and evaluation pipeline with three fixed random seeds.

For each seed:

1. Reproduce the data splits.
2. Train a fresh CNN.
3. Fit temperature scaling on the calibration split.
4. Fit standard and class-conditional conformal thresholds.
5. Evaluate every method on clean and shifted scenarios.
6. Evaluate all abstention thresholds.
7. Save run-level metrics.

Aggregate results across seeds using:

- Mean
- Standard deviation
- 95% confidence interval for the mean using the Student t distribution

Where useful, compare clean and shifted results using paired differences by seed. Do not present significance tests as meaningful when only a few observations are available. Emphasize effect sizes, confidence intervals, and visible trends.

---

## 13. Required Experiments

### Experiment A: Clean baseline

Compare raw and temperature-scaled predictions on the clean test set.

Purpose:

- Establish classification performance.
- Measure initial miscalibration.
- Verify that temperature scaling changes confidence metrics but not accuracy.

### Experiment B: Shift robustness

Evaluate raw and calibrated probabilities across every shift and severity.

Purpose:

- Measure how accuracy and confidence quality change.
- Identify cases where confidence remains high despite increasing error.

### Experiment C: Conformal coverage

Evaluate standard conformal prediction at every alpha value on clean and shifted data.

Purpose:

- Verify approximate clean-data coverage.
- Quantify coverage loss under distribution shift.
- Measure how set sizes change with severity.

### Experiment D: Class-conditional comparison

Compare standard and class-conditional conformal prediction.

Purpose:

- Compare marginal coverage with worst-class coverage.
- Measure the additional prediction-set size needed for class-level balance.

### Experiment E: Abstention trade-off

Evaluate confidence, singleton-set, and hybrid abstention.

Purpose:

- Produce risk-coverage curves.
- Determine how much accuracy improves when fewer predictions are accepted.
- Choose one understandable policy for the upload demo.

### Experiment F: Real-image examples

Collect a small demonstration set of approximately 20 personally sourced clothing photographs covering several Fashion-MNIST categories. Labels may be recorded manually for demonstration purposes.

Purpose:

- Show the train-to-real-image domain mismatch.
- Compare original photographs with their processed 28 x 28 versions.
- Inspect confidence, conformal sets, and abstention decisions.

This is a qualitative case study, not a statistically representative benchmark. Do not scrape or build a second large dataset.

---

## 14. Experiment Storage

Use SQLite only for structured experiment tracking. Keep the schema small.

### `models`

- `model_id`
- `seed`
- `config_hash`
- `checkpoint_path`
- `best_epoch`
- `validation_loss`
- `validation_accuracy`
- `temperature`
- `created_at`

### `scenarios`

- `scenario_id`
- `shift_name`
- `severity_name`
- `parameters`
- `seed`
- `sample_count`

The `parameters` field may contain a short serialized configuration because it is experiment metadata, not an event-log system.

### `runs`

- `run_id`
- `model_id`
- `scenario_id`
- `method_name`
- `alpha`
- `abstention_policy`
- `abstention_threshold`
- `started_at`
- `completed_at`
- `status`

### `metrics`

- `metric_id`
- `run_id`
- `metric_name`
- `class_name`, nullable
- `metric_value`

SQLite is not required for uploaded images. Uploaded images should be processed in memory and discarded after the response.

---

## 15. CSV Outputs

Produce tidy CSV files that can be inspected without the application:

### `run_metrics.csv`

One row per seed, scenario, method, alpha, abstention policy, and metric.

### `aggregate_metrics.csv`

One row per aggregated comparison with mean, standard deviation, confidence-interval lower bound, and confidence-interval upper bound.

### `per_class_metrics.csv`

One row per seed, scenario, method, alpha, class, and per-class metric.

### `reliability_bins.csv`

One row per scenario, method, seed, and calibration bin.

### `risk_coverage.csv`

One row per seed, scenario, abstention policy, threshold, coverage, and selective risk.

Do not create verbose event logs, replay files, or duplicate result formats.

---

## 16. Required Charts

Generate static PNG charts from saved CSV results:

1. Training and validation loss curves
2. Clean-data confusion matrix
3. Reliability diagram before and after temperature scaling
4. Accuracy against shift severity
5. Expected calibration error against shift severity
6. Conformal coverage against shift severity with a target-coverage reference line
7. Average conformal-set size against shift severity
8. Worst-class coverage: standard versus class-conditional conformal prediction
9. Risk-coverage curves for abstention policies
10. A compact comparison chart of clean versus strongest-shift results

Every chart must include a title, labelled axes, legend when needed, and uncertainty bars for aggregated multi-seed results when appropriate.

---

## 17. Suggested Project Structure

```text
clothsense/
|-- backend/
|   |-- app.py
|   |-- schemas.py
|   `-- inference_service.py
|-- frontend/
|   |-- src/
|   |   |-- components/
|   |   |-- pages/
|   |   |-- api/
|   |   `-- App.jsx
|   `-- package.json
|-- clothsense/
|   |-- config.py
|   |-- data.py
|   |-- model.py
|   |-- training.py
|   |-- calibration.py
|   |-- conformal.py
|   |-- abstention.py
|   |-- shifts.py
|   |-- metrics.py
|   |-- experiments.py
|   |-- storage.py
|   |-- plotting.py
|   `-- upload_preprocessing.py
|-- configs/
|   |-- training.yaml
|   `-- experiments.yaml
|-- data/
|-- artifacts/
|   |-- models/
|   |-- plots/
|   `-- processed_examples/
|-- results/
|   |-- run_metrics.csv
|   |-- aggregate_metrics.csv
|   |-- per_class_metrics.csv
|   |-- reliability_bins.csv
|   `-- risk_coverage.csv
|-- tests/
|-- scripts/
|   |-- train.py
|   |-- run_experiments.py
|   |-- export_results.py
|   `-- generate_plots.py
|-- research_report.md
|-- README.md
`-- requirements.txt
```

The exact module boundaries may change during implementation, but training, calibration, conformal prediction, shifts, evaluation, storage, API logic, and frontend code must remain logically separated.

---

## 18. Command-Line Workflows

Provide simple commands or scripts for these actions:

```text
train one model seed
train all required seeds
run clean evaluation
run all shift experiments
export CSV results
generate all charts
start local API
start local React frontend
```

The experiment runner must read configuration rather than requiring source-code edits for seeds, alpha values, shifts, severities, calibration bins, or abstention thresholds.

---

## 19. Testing Requirements

Implement focused tests for research-critical behavior:

- Dataset splits are disjoint, stratified, and reproducible.
- The CNN returns logits with shape `[batch_size, 10]`.
- Temperature is positive.
- Temperature scaling does not change argmax predictions.
- Conformal thresholds use only calibration data.
- Conformal sets contain only valid class IDs.
- Class-conditional thresholds are calculated from the correct classes.
- Every shift preserves shape and valid pixel range.
- Shift generation is deterministic for a fixed seed.
- Metric functions match small hand-calculated examples.
- Abstention counts accepted and rejected predictions correctly.
- Image upload validation rejects unsupported or malformed files.
- Upload preprocessing always produces the expected tensor shape.
- API responses satisfy their declared schema.

Use small fixtures and subsets in tests so the test suite does not retrain full models.

---

## 20. Implementation Phases

## Phase 1: Repository and Configuration Foundation

### Goal

Create the project structure and establish reproducibility before implementing machine learning.

### Steps

1. Create the Python package, scripts, tests, configuration, artifacts, and results directories.
2. Create a Python virtual environment and install the minimal dependencies: PyTorch, torchvision, NumPy, pandas, scikit-learn, SciPy, Pillow, Matplotlib, Seaborn, PyYAML, FastAPI, Uvicorn, and pytest.
3. Define central configuration for paths, seeds, training settings, alpha values, shift parameters, calibration bins, and abstention thresholds.
4. Add utilities that seed Python, NumPy, and PyTorch.
5. Add a device selector that uses CUDA when available and CPU otherwise.
6. Create a minimal SQLite initialization function.
7. Add a smoke test that imports the package and loads configuration.

### Completion criteria

- The package imports successfully.
- Configuration loads from one known location.
- Output directories are created safely.
- Repeated runs can use the same explicit seed.
- The initial automated tests pass.

---

## Phase 2: Dataset Pipeline

### Goal

Create reproducible, leakage-free data splits and loaders.

### Steps

1. Download Fashion-MNIST through torchvision.
2. Inspect class counts, image shapes, pixel range, and labels.
3. Generate stratified train, validation, and calibration indices from the official training set.
4. Save the split indices so all later stages reuse them.
5. Define training transforms and deterministic evaluation transforms.
6. Calculate or confirm the training normalization values and store them in configuration.
7. Build separate data loaders for training, validation, calibration, and clean testing.
8. Create a small dataset-summary table and a grid of sample images for verification.
9. Test split disjointness, reproducibility, class balance, and tensor shapes.

### Completion criteria

- Every example belongs to exactly one intended split.
- Calibration and test data never enter CNN training.
- Re-running with the same seed recreates the same splits.
- Data loaders return correctly normalized tensors and labels.

---

## Phase 3: CNN Training and Baseline Evaluation

### Goal

Train a reliable but compact classifier and establish its clean-data baseline.

### Steps

1. Implement the CNN architecture.
2. Implement training and validation loops.
3. Record epoch-level loss and accuracy.
4. Add early stopping based on validation loss.
5. Save the best checkpoint rather than the last checkpoint.
6. Train an initial development seed and inspect learning curves.
7. Fix obvious data, convergence, or overfitting problems without performing a large hyperparameter search.
8. Freeze the training configuration.
9. Train the three required seeds.
10. Evaluate the selected checkpoints on the clean test set.
11. Store model metadata and baseline metrics.

### Completion criteria

- Training is reproducible for a fixed seed within normal hardware limits.
- Best checkpoints reload correctly.
- Clean accuracy is reasonable for a small Fashion-MNIST CNN.
- Training curves, confusion matrix data, and baseline metrics are saved.

---

## Phase 4: Calibration, Conformal Prediction, and Abstention

### Goal

Add all uncertainty methods while preserving strict separation between calibration and test data.

### Steps

1. Run each trained model over the calibration split and save logits and labels.
2. Implement raw softmax prediction.
3. Implement positive scalar temperature scaling.
4. Fit one temperature per trained model seed using calibration negative log-likelihood.
5. Verify that temperature scaling leaves argmax class predictions unchanged.
6. Implement standard split-conformal scores and finite-sample quantiles for every alpha.
7. Implement class-conditional thresholds for every class and alpha.
8. Implement prediction-set construction.
9. Implement confidence-threshold, singleton-set, and hybrid abstention policies.
10. Add unit tests using small manually constructed probability arrays.
11. Save learned temperatures and conformal thresholds with their corresponding model artifacts.

### Completion criteria

- Every uncertainty method is deterministic for fixed inputs.
- All fitted values come only from the calibration split.
- Standard and class-conditional prediction sets are generated correctly.
- Abstention policies return both a decision and an interpretable reason.

---

## Phase 5: Shift Generation and Metrics

### Goal

Build the complete controlled evaluation environment.

### Steps

1. Implement Gaussian noise at all severities.
2. Implement rotation at all severities.
3. Implement blur at all severities.
4. Implement brightness and contrast scenarios.
5. Implement deterministic class-imbalance subsampling.
6. Add scenario objects containing names, parameters, seeds, and sample counts.
7. Visually inspect a fixed sample grid for every image corruption and severity.
8. Implement classification metrics.
9. Implement probability and calibration metrics.
10. Implement conformal coverage and set-size metrics.
11. Implement per-class and worst-class metrics.
12. Implement selective-prediction metrics and risk-coverage data.
13. Validate metric functions against small hand-calculated cases and trusted library outputs where available.

### Completion criteria

- All shifts preserve labels, tensor shapes, and valid pixel ranges.
- Stronger severities visibly create stronger corruptions.
- Metrics handle empty accepted sets and empty conformal sets safely.
- Per-class and aggregate outputs use consistent names and schemas.

---

## Phase 6: Experiment Runner, SQLite, and CSV Results

### Goal

Run the full experiment matrix reproducibly and save compact, queryable results.

### Steps

1. Implement SQLite tables for models, scenarios, runs, and metrics.
2. Implement safe insertion and retrieval helpers.
3. Create a configuration-driven experiment matrix across seeds, scenarios, methods, alpha values, and abstention policies.
4. Avoid redundant model inference by calculating logits once per model and scenario, then reusing them for all uncertainty methods.
5. Record run status so an interrupted run can be identified.
6. Run a small subset as a smoke experiment.
7. Validate row counts, foreign-key relationships, metric ranges, and missing values.
8. Run the full required experiment matrix.
9. Export the five required tidy CSV files.
10. Aggregate repeated-seed results with mean, standard deviation, and 95% confidence intervals.
11. Add a command that rebuilds aggregate CSV files from SQLite without rerunning inference.

### Completion criteria

- One command can run the configured experiment matrix.
- Failed or incomplete runs are visible rather than silently treated as successful.
- SQLite and CSV values agree for sampled records.
- Results can be filtered by seed, shift, severity, method, alpha, class, and abstention policy.
- No JSON event-log or replay subsystem exists.

---

## Phase 7: Analysis and Research Figures

### Goal

Turn saved results into a defensible research analysis.

### Steps

1. Generate every required chart from CSV files rather than from hidden in-memory values.
2. Compare raw and calibrated confidence on clean data.
3. Quantify changes in accuracy, NLL, Brier score, and ECE as shift severity increases.
4. Compare empirical conformal coverage with nominal target coverage.
5. Compare standard and class-conditional conformal methods on marginal and worst-class coverage.
6. Compare prediction-set size across methods and shifts.
7. Generate risk-coverage curves for all abstention policies.
8. Select one alpha and one abstention policy for the upload demonstration based on a clear trade-off, not only the highest observed accuracy.
9. Identify at least three informative failure cases, such as high-confidence errors, coverage collapse, or very large prediction sets.
10. Save a compact summary table of the main findings.

### Completion criteria

- Every central research question has at least one corresponding result.
- Figures are regenerated entirely from saved result files.
- Claims distinguish clean-data observations from shifted-data observations.
- The limitations of finite seeds and Fashion-MNIST are stated accurately.

---

## Phase 8: Upload Inference Service and Project Report

### Goal

Prepare the stable inference logic and written project explanation before building the interface.

### Steps

1. Implement and test the real-image preprocessing pipeline.
2. Add optional inversion as an explicit request setting.
3. Load one selected model, its temperature, and its conformal thresholds through a reusable inference service.
4. Return raw, calibrated, conformal, and abstention outputs from one service method.
5. Assemble approximately 20 personal demonstration photographs if practical.
6. Run the demonstration set through the inference service and save only the chosen examples needed for the report.
7. Write `research_report.md` with the objective, method, experiment design, results, failure cases, limitations, and conclusion.
8. Write a concise `README.md` containing setup, commands, project structure, and how to reproduce the main experiment.
9. Verify every reported number against the final CSV files.

### Completion criteria

- A local Python call can classify an uploaded-style image without the frontend.
- The inference service loads artifacts once and reuses them.
- The report is supported by saved metrics and figures.
- The README is sufficient for a professor to run the core project locally.

---

## Phase 9: React Frontend and Minimal Local API

### Goal

Create the final presentation layer after all research and inference logic is complete.

### 9.1 FastAPI backend

Implement only these endpoints:

#### `GET /api/classes`

Return the ten supported class IDs and names.

#### `POST /api/classify`

Accept one image and simple options such as inversion and alpha. Return the inference fields defined in the upload workflow. Process the image in memory and do not retain it after the response.

#### `GET /api/results/summary`

Return a small prepared subset of aggregate results needed by the dashboard.

#### `GET /api/results/charts`

Return the names or local URLs of generated research charts.

The API must call the existing Python modules. Do not duplicate calibration, preprocessing, conformal, metric, or model code inside endpoint functions.

### 9.2 React pages

Create three pages.

#### Upload and Classify

Include:

- Drag-and-drop area and file picker
- Original-image preview
- Processed 28 x 28 preview enlarged with nearest-neighbour display
- Invert-image toggle
- Alpha selector
- Classify button
- Predicted label
- Raw and calibrated confidence
- Top-three probability bars
- Standard conformal set
- Class-conditional conformal set
- Accept or uncertain decision
- Plain-language decision reason
- Visible note explaining the Fashion-MNIST domain limitation

#### Experiment Results

Include:

- A few top-level metric cards
- Filters for shift, severity, method, and alpha
- Reliability diagram
- Coverage-versus-severity chart
- Set-size-versus-severity chart
- Risk-coverage chart
- Compact aggregated-results table

Use the already generated result files. The frontend must not launch model training or the full experiment suite.

#### Example Gallery

Include:

- A small set of selected Fashion-MNIST and personal-photo examples
- Original image when available
- Processed image
- True or expected class when known
- Prediction and confidence
- Conformal set
- Abstention decision
- A few deliberately selected failure cases

### 9.3 Interface constraints

- Use React with a simple Vite setup.
- Use ordinary CSS or one lightweight component library.
- Keep the interface desktop-friendly and responsive enough for a laptop browser.
- Do not add authentication, accounts, settings management, animations, themes, or admin pages.
- Do not add deployment configuration.
- Keep charts readable and explanations brief.

### 9.4 Final verification

1. Start the API and frontend locally.
2. Verify all three pages load without console errors.
3. Upload valid JPG and PNG images.
4. Verify invalid and oversized files show useful errors.
5. Confirm displayed values match direct Python inference for the same image.
6. Confirm chart and summary values match the final CSV files.
7. Confirm uploaded images are not stored.
8. Run the automated test suite one final time.
9. Follow the README from a clean local session and correct missing steps.

### Completion criteria

- A professor can inspect the research results and classify a photograph locally.
- The original and processed images make the domain transformation visible.
- The interface clearly distinguishes prediction, confidence, conformal uncertainty, and abstention.
- The application uses saved model and result artifacts and performs no retraining.
- No deployment work is included.

---

## 21. Final Acceptance Criteria

The project is complete only when:

- One CNN is trained for all required seeds using leakage-free splits.
- Temperature scaling, standard conformal prediction, class-conditional conformal prediction, and abstention are implemented and tested.
- All clean and shifted experiments have been completed.
- Metrics are stored in SQLite and exported to the required CSV files.
- Aggregate results include uncertainty across seeds.
- Required research charts are generated from saved results.
- The upload pipeline accepts a real clothing photograph and shows its processed representation.
- The local React interface exposes the classifier and selected experiment results.
- The interface warns that uploaded photographs may differ substantially from Fashion-MNIST.
- The report states both successful findings and failure cases.
- Tests pass and the reproduction instructions have been checked.
- No deployment, authentication, commercial, or unnecessary production features are present.

---

## 22. Definition of a Strong Final Demonstration

The final demonstration should take approximately five minutes:

1. Show the clean-data accuracy and reliability diagram.
2. Show how calibration error and conformal coverage change as one shift becomes stronger.
3. Compare standard and class-conditional worst-class coverage.
4. Show a risk-coverage curve and explain why abstention can improve accepted-prediction accuracy.
5. Upload one easy clothing photograph and inspect its processed image and uncertainty results.
6. Upload one difficult photograph and show the system abstaining or returning multiple plausible classes.
7. End with the central conclusion: confidence methods can improve uncertainty reporting on familiar data, but their guarantees and reliability can weaken when the data distribution changes.
