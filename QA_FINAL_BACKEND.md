# Final backend QA handoff

Verify independently:

- deterministic 45k/5k/10k training splits, official-test separation, class
  stratification, saved-index reuse, and absence of leakage;
- seeded initialization/DataLoader behavior, fixed CNN architecture, validation-loss
  checkpoint selection, and three independent checkpoints;
- calibration-split-only temperature/conformal fitting, temperature argmax
  invariance, finite-sample conformal quantiles, and class-threshold indexing;
- shift determinism, label/index preservation, raw-space corruption, normalization,
  and absence of shifted-data retraining or recalibration;
- classification, calibration, conformal, selective-risk, empty-bin, zero-acceptance,
  and runtime metric formulas;
- 45-scenario/1,575-run matrix completeness, inference reuse, run-state uniqueness,
  SQLite foreign keys, CSV agreement, incomplete-run exclusion, and three-seed
  Student-t confidence intervals;
- regeneration of all ten figures from persisted CSV files only;
- provenance of seed 73 and the configured alpha 0.10 hybrid/0.80 demo policy;
- EXIF orientation, format/size validation, square padding, grayscale resize,
  inversion, Phase 1 normalization, and exact processed-image representation;
- checkpoint/uncertainty loading once, no fitting during inference, no upload
  persistence, and direct-Python versus `/api/classify` equality;
- `/api/classes`, `/api/classify`, `/api/results/summary`, chart URLs, and useful
  client errors without stack traces;
- every number in `research_report.md` against final CSVs and all README commands
  from a clean environment;
- scope exclusions: no frontend, deployment, authentication, upload history,
  additional dataset, OOD detector, or API-triggered training/calibration.
