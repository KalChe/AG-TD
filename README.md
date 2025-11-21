# ag-td

adversarially-guided td for provable bounds

## structure

- `src/` - core modules (environment, networks, challenger, algorithms)
- `experiments/` - training and evaluation code
- `results/` - generated figures and tables

## setup

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## run

```bash
python main.py
```

runs full experiments.

## citation

```bash
@inproceedings{
cherukuri2025adversariallyguided,
title={Adversarially-Guided {TD}: Learning Robust Value Functions with Counter-Example Replay},
author={Kalyan Cherukuri},
booktitle={Workshop on Differentiable Learning of Combinatorial Algorithms},
year={2025},
url={https://openreview.net/forum?id=eYV3Ijpdrg}
}
```
