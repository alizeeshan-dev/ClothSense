export const chartInterpretations = {
  '01_training_validation_loss.png': 'Training and validation loss track convergence for the three fixed model seeds.',
  '02_clean_confusion_matrix.png': 'The clean confusion matrix shows which Fashion-MNIST classes remain difficult to separate.',
  '03_reliability_raw_vs_temperature.png': 'Temperature scaling brings clean-data confidence closer to observed accuracy.',
  '04_accuracy_vs_shift_severity.png': 'Accuracy generally falls as noise, rotation, and blur become more severe.',
  '05_ece_vs_shift_severity.png': 'Calibration quality can deteriorate sharply even when confidence remains high.',
  '06_conformal_coverage_vs_shift_severity.png': 'Nominal conformal coverage is approximately met on clean data but weakens under strong shifts.',
  '07_conformal_set_size_vs_shift_severity.png': 'Prediction-set size reveals how conformal informativeness changes with corruption.',
  '08_worst_class_coverage.png': 'Class-conditional thresholds improve clean worst-class coverage at a modest set-size cost.',
  '09_risk_coverage.png': 'Abstention trades the fraction of accepted predictions for lower error among those accepted.',
  '10_clean_vs_strongest_shift.png': 'The strongest shifts expose a clear gap between familiar-data and shifted-data behavior.',
}

export const chartTopics = {
  '01_training_validation_loss.png': 'model',
  '02_clean_confusion_matrix.png': 'model',
  '03_reliability_raw_vs_temperature.png': 'calibration',
  '04_accuracy_vs_shift_severity.png': 'shift',
  '05_ece_vs_shift_severity.png': 'calibration',
  '06_conformal_coverage_vs_shift_severity.png': 'conformal',
  '07_conformal_set_size_vs_shift_severity.png': 'conformal',
  '08_worst_class_coverage.png': 'conformal',
  '09_risk_coverage.png': 'selective',
  '10_clean_vs_strongest_shift.png': 'shift',
}
