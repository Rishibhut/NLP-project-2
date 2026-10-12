# Experiment Results & Evaluation Logs

This directory contains reproducible evaluation outputs, experiment logs, confusion matrices, and alignment gap metrics exported by the Hinglish Product Review Analyzer.

## Export File Naming Conventions
- `experiment_<timestamp>_<model_name>.json`: Experiment hyperparameter configuration, data split seeds, and computed metrics.
- `gap_analysis_<timestamp>.csv`: Subset evaluation records contrasting same-language vs. cross-lingual alignment F1 scores.
- `error_analysis_<timestamp>.csv`: Detailed per-instance prediction logs with categorized error diagnostics.
