# GardDT

An exploratory gradient-based decision-tree implementation in PyTorch, alongside GradTree reference code.

## Repository contents

- `GTpytorch.py`: experimental `GTpytorch` / `GradTreeBlock` implementation, training loop, and preprocessing placeholder.
- `test.py`: development experiment using OpenML dataset 40536, ordinal encoding, and train/validation/test splits.
- `GradTree/` and `GradTree-master/`: reference material kept separately from the PyTorch experiment; see the original attribution and license files inside these directories.

## Current status

This is an **unfinished research prototype**, not a validated implementation or production library. The preprocessing method is a placeholder. The experiment calls `model.predict()`, which is not implemented in `GTpytorch.py`, and its binary objective/output shapes need review. No benchmark or model-parity claim is made.

## Inspecting the experiment

The scripts import PyTorch, pandas, NumPy, scikit-learn, category-encoders, and OpenML. `test.py` downloads data from OpenML and is an experiment script, not a passing automated test suite. Review the implementation limitations above before attempting to run it.

For my current data engineering, ML, and application work, see [my portfolio](https://whwan4570.github.io/).
