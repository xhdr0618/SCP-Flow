# SCP-Flow

Structure-Controllable Progression Flow for longitudinal glaucoma forecasting from fundus image sequences.

This repository starts from the official `tHPM-LDM` codebase and extends it into a flow-based forecasting framework named `SCP-Flow`. The current implementation keeps the original historical and population conditioning modules, while replacing the future latent diffusion predictor with a progression flow predictor in latent space.

## Overview

SCP-Flow is designed for longitudinal glaucoma forecasting with the following capabilities:

- history-conditioned future latent prediction
- latent future fundus image generation
- structure-aware conditioning with:
  - `vCDR`
  - `OC-OD`
  - `disc ROI`
  - `polar ROI`
- next-visit interval prediction
- uncertainty prediction
- progression consistency regularization
- missing-visit robustness evaluation

The current repository supports two predictor paths:

- `diffusion`: the original `tHPM-LDM` latent diffusion path
- `scpflow`: the new flow-based progression forecasting path

## Current Status

The current `scpflow` implementation is a minimum runnable research prototype.

Already implemented:

- SCP-Flow backbone
- reuse of `t-MSHF` and `PMQM`
- frozen first-stage VQGAN encoder/decoder
- structure condition interface
- interval head
- uncertainty head
- consistency loss
- missing-visit evaluation utilities
- LaTeX paper draft skeleton

Still incomplete:

- full-scale long training
- strong uncertainty calibration
- clinically validated structure supervision
- multi-step rollout
- formal benchmark against the original diffusion path under matched settings

## Repository Structure

Main code:

- `train_vqldm_PMN.py`: main training and testing entry
- `models/flow_forecaster.py`: SCP-Flow predictor
- `datamodule/seq_fundus_2D_datamodule.py`: longitudinal SIGF dataloader
- `metrics/evaluate_scpflow.py`: SCP-Flow evaluation
- `metrics/run_scpflow_missing_visit_sweep.py`: missing-visit sweep
- `metrics/aggregate_scpflow_missing_visit_results.py`: robustness aggregation

Documentation:

- `SCP_FLOW_PROGRESS.md`: project progress tracker
- `SCP_FLOW_METHODS.md`: current method description
- `latex_paper/`: English LaTeX paper framework

Original base modules still reused:

- `ldm/`
- `networks/tMSHF/`
- `networks/PMQM/`
- `train_vqgan.py`
- `train_classifier.py`

## Data

This repository assumes the processed SIGF dataset is prepared as `SIGF_make`:

```text
data/
  SIGF_make/
    train/
    validation/
    test/
```

Each sample is an `.npz` file containing:

- `seq_imgs`
- `times`
- `labels`

Large datasets, checkpoints, and outputs are excluded from Git tracking via `.gitignore`.

## Environment

The original environment files are kept under:

- `envs/requirements.txt`
- `envs/environment.yml`

You can also adapt the original environment setup from the inherited `tHPM-LDM` project structure.

## Training

### PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_scpflow.ps1
```

### Bash

```bash
bash scripts/train_scpflow.sh
```

These scripts launch the current recommended single-step SCP-Flow training setup.

## Testing

### PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File scripts\test_scpflow.ps1 -CheckpointPath "path_to_checkpoint.ckpt"
```

### Bash

```bash
bash scripts/test_scpflow.sh path_to_checkpoint.ckpt
```

The test path exports:

- predicted future images: `*_gen.png`
- ground-truth future images: `*_gt.png`
- reconstructed target images: `*_rec.png`
- structured predictions: `flow_predictions.csv`

## Evaluation

### PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File scripts\metric_scpflow.ps1 -EvalPath "results/test_scpflow"
```

### Bash

```bash
bash scripts/metric_scpflow.sh results/test_scpflow
```

The current evaluation includes:

- image quality metrics
- next-visit interval error
- uncertainty summary
- missing-visit robustness statistics

## Missing-Visit Robustness

The current implementation supports simulated missing visits with:

- `none`
- `tail`
- `uniform`
- `random`

Useful scripts:

- `metrics/run_scpflow_missing_visit_sweep.py`
- `metrics/aggregate_scpflow_missing_visit_results.py`

## LaTeX Paper Draft

An English LaTeX paper skeleton is available under:

- `latex_paper/main.tex`

It already includes:

- introduction
- related work
- methods
- experiments
- discussion
- conclusion
- placeholder tables
- bibliography file

## Notes

- The current `vCDR` and `OC-OD` values are heuristic proxies derived from ROI extraction, not manually curated clinical annotations.
- The uncertainty head is implemented, but its calibration is still under development.
- The current best-verified SCP-Flow runs are prototype-level rather than final benchmark-quality experiments.

## Citation

If you use this repository, please cite the final SCP-Flow paper once available. Until then, please cite the original tHPM-LDM paper if you use the inherited base components.

## Acknowledgements

This repository builds on the original `tHPM-LDM` codebase and also inherits ideas or components from related open-source projects in latent generative modeling, retinal image forecasting, and longitudinal trajectory modeling.
