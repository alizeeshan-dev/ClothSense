# ClothSense: uncertainty under image-distribution shift

## Objective

ClothSense studies whether the uncertainty reported by a compact clothing-image
classifier remains trustworthy as inputs move away from its Fashion-MNIST training
distribution. The project compares raw softmax confidence, temperature scaling,
split-conformal prediction sets, class-conditional conformal sets, and abstention.
It is a local research prototype, not a reliable real-world clothing-recognition
system.

All numerical results below come from `results/run_metrics.csv`,
`results/aggregate_metrics.csv`, `results/per_class_metrics.csv`,
`results/reliability_bins.csv`, or `results/risk_coverage.csv`. Reported central
values are means over the three fixed seeds unless stated otherwise.

## Dataset and experimental design

The official Fashion-MNIST training set was split deterministically and stratified
into 45,000 training, 5,000 validation, and 10,000 calibration examples. The
official 10,000-image test set remained separate. Model parameters used only the
training split; validation loss selected checkpoints; temperature and conformal
thresholds used only calibration data; test labels were used only for evaluation.

Fresh models were trained with seeds 17, 42, and 73. Every model was evaluated on
clean data and fourteen controlled scenarios: three Gaussian-noise levels, three
rotations, three Gaussian-blur levels, three brightness/contrast settings, and two
class-imbalance levels. Within a seed, one inference pass supplied logits to every
uncertainty method for a scenario. No model was retrained or recalibrated on shifted
data.

## Model and training

The fixed CNN has 32- and 64-channel 3×3 convolutional layers, ReLU activations,
2×2 max pooling, a 128-unit fully connected layer, dropout 0.50, and ten output
logits. Training used cross-entropy, Adam with learning rate 0.001, batch size 128,
at most 20 epochs, and early stopping after three non-improving validation epochs.
The best checkpoint for each seed was selected by validation loss. Seed 73 is used
for upload inference because it had the lowest validation loss, not because of
test-set performance.

## Uncertainty methods

Raw softmax provides the uncalibrated class probabilities. Temperature scaling fits
one positive scalar per model on calibration logits and leaves the argmax unchanged.
Standard split conformal uses calibrated true-class scores and a finite-sample
corrected quantile. Class-conditional conformal fits the same type of threshold
separately within each true class. Results were evaluated at alpha 0.05, 0.10, and
0.20. Confidence, singleton-set, and hybrid abstention policies were evaluated over
the configured confidence thresholds.

## Experiments and results

Experiment A established the clean baseline. Mean clean accuracy was 0.9173. On
clean data, temperature scaling reduced ECE from 0.0159 to 0.0076 and NLL from
0.2426 to 0.2329 without changing predicted classes.

Experiment B measured shift robustness. The strongest configured corruptions had
the following results at alpha 0.10:

| Scenario | Accuracy | Calibrated ECE | Standard coverage | Mean set size |
|---|---:|---:|---:|---:|
| Gaussian noise, strong | 0.4262 | 0.3702 | 0.3889 | 0.870 |
| Rotation, strong | 0.3270 | 0.2891 | 0.2351 | 0.580 |
| Gaussian blur, strong | 0.7548 | 0.0795 | 0.6145 | 0.700 |
| Brightness reduction, strong | 0.8890 | 0.0106 | 0.8528 | 0.924 |
| Class imbalance, strong | 0.9407 | 0.0093 | 0.9263 | 0.968 |

The class-imbalance accuracy increase does not imply a better classifier: the
retained distribution emphasizes configured majority classes, so per-class and
macro results remain necessary.

Experiment C evaluated conformal coverage. On clean data, standard conformal
coverage was 0.9493, 0.8971, and 0.7955 for nominal coverage levels 0.95, 0.90, and
0.80. At alpha 0.10, the clean mean set size was 0.957. Coverage weakened sharply
under strong noise and rotation, showing that calibration-split coverage guarantees
do not transfer unchanged to shifted distributions.

Experiment D compared class-conditional conformal prediction. At alpha 0.10,
clean worst-class coverage improved from 0.6627 with standard conformal to 0.8777
with class-conditional conformal. Mean set size increased from 0.957 to 1.012. This
clean-data improvement did not create shift robustness: class-conditional
worst-class coverage reached 0.0000 under strong rotation.

Experiment E examined abstention. The selected demonstration policy uses alpha
0.10, a hybrid singleton-and-confidence rule, and threshold 0.80. On clean data it
accepted 0.8228 of examples with selective risk 0.0230. Averaged equally across all
configured scenarios and seeds, acceptance was 0.6519 and selective risk 0.1003.
The setting was chosen as an interpretable balance, rather than selecting a policy
that achieved low risk by accepting almost nothing.

Experiment F is prepared but not numerically reported because no personal-photo set
was available in the repository. Once supplied, those photographs are processed as
a qualitative domain-mismatch case study, not a representative benchmark.

## Informative failure patterns

1. Under strong Gaussian noise, raw mean confidence remained 0.8429 while accuracy
   fell to 0.4262. Temperature scaling reduced mean confidence to 0.7963 but could
   not correct the distribution shift; calibrated ECE was 0.3702.
2. Strong rotation reduced alpha-0.10 standard conformal coverage to 0.2351,
   far below the nominal 0.90 target. The mean set size was 0.580 and the empty-set
   rate was 0.4197, showing that shifted inputs can produce unhelpfully small or
   empty sets, not only broad sets.
3. Standard conformal achieved 0.8971 marginal clean coverage at alpha 0.10 while
   its worst-class coverage was only 0.6627. Class-conditional calibration improved
   that clean disparity, but its worst-class coverage still collapsed under severe
   shifts.
4. Ordinary photographs have backgrounds, texture, pose, and lighting unlike
   centered Fashion-MNIST images. Confidence, set size, and abstention can describe
   model uncertainty, but none is a proven out-of-distribution detector.

## Limitations

The study uses one small CNN and one dataset, only three seeds, and synthetic
controlled shifts. Confidence-bin estimates and Student-t intervals are therefore
descriptive rather than broad evidence of generalization. Class-conditional
calibration has finite samples per class. Conformal guarantees assume exchangeable
calibration and evaluation data and need not hold after distribution shift.
Personal photographs, when added, are qualitative examples only. Abstention may
reduce accepted-sample error, but it is not an OOD detector.

## Conclusion

On familiar Fashion-MNIST data, temperature scaling improved calibration,
conformal sets approximately met their marginal targets, class-conditional fitting
improved worst-class coverage, and abstention exposed a useful risk–coverage
trade-off. Under stronger distribution shifts, classification accuracy,
calibration, and conformal coverage could deteriorate substantially. These methods
can communicate uncertainty more honestly, but their reliability and guarantees
must not be assumed to survive a change in data distribution.
