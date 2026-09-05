# QA handoff: Phases 1–2

- Verify persisted splits are disjoint, stratified, reproducible, and tied to split seed 2027.
- Confirm training uses only the 45,000 training indices and validation-only checkpoint selection.
- Confirm calibration is untouched and clean test is evaluated only after checkpoint selection.
- Verify the CNN layers and raw-logit output exactly match the project plan.
- Re-run one seed to assess expected same-device seeded reproducibility.
- Confirm every checkpoint is the minimum-validation-loss epoch and reloads independently.
- Cross-check checkpoint, history JSON, clean-output NPZ, SQLite row, and config hash per seed.
- Recompute clean accuracy, macro F1, per-class metrics, and confusion matrix from saved outputs.

