# RL Chinese Checkers

Reinforcement learning project for Chinese Checkers using a custom environment, heuristic opponents, and Stable-Baselines3 (MaskablePPO).

This project is designed to:

- Train agents in multi-phase self-play and heuristic-opponent settings.
- Visualize learning progress with training metric plots.
- Replay trained models in a GUI.
- Generate board heatmaps that show where a model tends to place pieces over time.

## Requirements

- Python 3.10+ recommended
- Windows, macOS, or Linux
- Tkinter support (needed for GUI scripts)

Install dependencies:

```bash
pip install -r requirements.txt
```

For plotting scripts, also install:

```bash
pip install pandas matplotlib
```

## Quick Start

From the project root folder:

1. Create and activate a virtual environment (recommended).

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install packages.

```bash
pip install -r requirements.txt
pip install pandas matplotlib
```

## How To Run

### 1) Train a Model

```bash
python "single system/train.py"
```

Default training behavior in `train.py`:

- Uses MaskablePPO with MLP policy.
- Rotates through predefined color phases.
- Uses VecNormalize statistics.
- Saves metrics when enabled.

### 2) Watch a Game in GUI

```bash
python "single system/play_gui.py"
```

By default, this script loads:

- `single system/saved_models/model1.zip`
- `single system/saved_models/model1.pkl`

If you trained your own model, update those paths in `play_gui.py` or copy your model files accordingly.

### 3) Plot Training Metrics

```bash
python "single system/plot_metrics.py"
```

This opens a matplotlib window with smoothed plots for:

- Episode reward
- Episode length
- Approximate KL
- Explained variance
- Value loss
- Entropy loss

### 4) Generate Heatmaps

```bash
python "single system/heatmap.py"
```

This script evaluates a saved model versus heuristics and writes PNGs into:

- `single system/heatmaps/`

Default snapshots are generated at turns 20, 40, and 60.
