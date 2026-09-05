# QA handoff: Phases 3–4

- Verify temperature and all conformal thresholds are fitted only from cached calibration logits/labels.
- Check scalar-temperature positivity, probability normalization, and argmax invariance.
- Independently verify finite-sample conformal ranks and alpha indexing.
- Confirm every class-conditional candidate uses its own class threshold without pooling.
- Check all shift parameters against configuration and reproduce image/subsample outputs by seed.
- Verify shifts preserve clean test tensors, labels, shape, and pre-normalization `[0,1]` range.
- Independently validate classification, calibration, conformal, and selective metric formulas.
- Check absent-class, empty-bin, empty-set, and zero-acceptance handling.
- Confirm shifted evaluation performs neither model training nor uncertainty refitting.

