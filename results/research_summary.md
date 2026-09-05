# ClothSense Phase 5 research summary

## Clean baseline

- Accuracy: 0.9173
- ECE, raw → temperature-scaled: 0.0159 → 0.0076
- NLL, raw → temperature-scaled: 0.2426 → 0.2329

## Strongest configured shifts

| Shift | Accuracy | Δ accuracy | Calibrated ECE | Standard coverage | Average set size |
|---|---:|---:|---:|---:|---:|
| gaussian_noise/strong | 0.4262 | -0.4911 | 0.3702 | 0.3889 | 0.870 |
| rotation/strong | 0.3270 | -0.5903 | 0.2891 | 0.2351 | 0.580 |
| gaussian_blur/strong | 0.7548 | -0.1625 | 0.0795 | 0.6145 | 0.700 |
| brightness_contrast/strong_brightness | 0.8890 | -0.0283 | 0.0106 | 0.8528 | 0.924 |
| class_imbalance/strong | 0.9407 | +0.0234 | 0.0093 | 0.9263 | 0.968 |

## Conformal and abstention trade-offs

- At alpha=0.1, clean worst-class coverage is 0.6627 for standard conformal and 0.8777 for class-conditional conformal.
- Demo policy `hybrid` has mean clean acceptance 0.8228 and selective risk 0.0230.

## Selected upload-demo configuration

- Conformal alpha: 0.1
- Abstention policy: hybrid
- Confidence threshold: 0.8
- Selection balances coverage and selective risk while retaining a conformal singleton condition that is straightforward to explain.
