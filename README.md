# C-MANET Reliability Assessment

Python simulations for evaluating cluster reliability in Cooperative MANETs (C-MANET) using Monte Carlo methods.

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Random Seed Configuration](#random-seed-configuration)
3. [Simulation Scripts Overview](#simulation-scripts-overview)
4. [Step-by-Step Figure Reproduction](#step-by-step-figure-reproduction)
5. [Key Parameters Reference](#key-parameters-reference)
6. [Troubleshooting](#troubleshooting)

---

## Environment Setup

### Required Software Versions

| Software | Version | Notes |
|----------|---------|-------|
| Python | 3.8 – 3.11 | 64-bit recommended |
| numpy | ≥ 1.20 | Numeric operations |
| matplotlib | ≥ 3.3 | Plotting and visualization |
| pandas | ≥ 1.0 | Data manipulation (for experiment scripts) |
| scipy | ≥ 1.7 | Statistical tests (for comparative experiments) |

### Installation

**Option 1: Using requirements.txt (Recommended)**

```bash
pip3 install -r requirements.txt
```

**Option 2: Manual installation**

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or: venv\Scripts\activate  # On Windows

# Install dependencies
pip3 install numpy>=1.20 matplotlib>=3.3 pandas>=1.0 scipy>=1.7
```

### Verify Installation

```bash
python -c "import numpy; import matplotlib; import pandas; import scipy; print('numpy:', numpy.__version__); print('matplotlib:', matplotlib.__version__); print('pandas:', pandas.__version__); print('scipy:', scipy.__version__)"
```

---

## Random Seed Configuration

All scripts use Python's built-in `random` and `numpy.random` modules. For reproducible results, add the following lines at the beginning of any script:

```python
import random
import numpy as np

# Set random seeds for reproducibility
RANDOM_SEED = 42  # Change this value for different random instances
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
```

**Note:** The current scripts do not hard-code random seeds to allow natural Monte Carlo variability. To reproduce exact figures, add the seed configuration above to the desired script.

---

## Simulation Scripts Overview

The scripts are organized into two categories:

### Core Reliability Assessment (C1, C2, C3)

| Script | Purpose | Key Outputs |
|--------|---------|-------------|
| `C1-Simulation of a Single Communication Task in C-MANET.py` | Simulates single source-to-target communication; computes time-varying path stability and transmission success | Network topology, communication path, P_stable(t), P_success(t) curves |
| `C2-Evaluation of Cluster Reliability in C-MANET.py` | Monte Carlo simulation (500 runs) for statistical evaluation of intra/inter-cluster reliability | Reliability distributions, histograms, convergence plots |
| `C3-Evaluation of Cluster Reliability Single Simulation.py` | Single-run evaluation of intra/inter-cluster reliability with exhaustive pair enumeration and cluster topology visualization | Cluster topology, reliability statistics |

### Experiment Scripts (E1, E2, E3)

| Script | Purpose | Key Outputs |
|--------|---------|-------------|
| `E1-Correlated-mobility experiments.py` | Sensitivity study: evaluates reliability across different R_ch_node values (50-95m) | CSV results, R_ch_node vs. reliability curves |
| `E2-interference-sensitivity experiment.py` | Monte Carlo sensitivity study on interference effects (R_ch_node: 50-95m) | CSV results, interference vs. reliability table |
| `E3-comparative experiment.py` | Comparative study: C-MANET vs. traditional MANET (no clustering) across R_ch_node values | CSV results, comparative reliability analysis |

---

## Step-by-Step Figure Reproduction

### Figure 1: Single Communication Task Visualization

**Script:** `C1-Simulation of a Single Communication Task in C-MANET.py`

**What it does:**
- Initializes a C-MANET with 50 nodes in a 100m × 100m area
- Selects cluster heads (12% of nodes)
- Finds a random source-to-target communication path
- Visualizes the network topology and communication path
- Plots path stability probability and transmission success rate over time

**Run command:**
```bash
python "C1-Simulation of a Single Communication Task in C-MANET.py"
```

**Expected outputs:**
1. **Figure window 1:** Network topology with:
   - Red triangles: Cluster heads
   - Light blue circles: Member nodes
   - Communication range circles (red for CHs, blue for members)
   - Blue dashed line: Communication path with hop annotations
   - Gold star: Source node
   - Green star: Target node

2. **Figure window 2:** Two subplots showing:
   - Left: Path Stability Probability P_stable(t) vs. Time
   - Right: Transmission Success Rate P_success(t) vs. Time

3. **Console output:**
   ```
   Initializing C-MANET with heterogeneous communication ranges...
   Selected task: Node X --> Node Y
   ✅ Path found:
     k1 (src→src_CH) = N
     k3 (CH↔CH)      = N
     k2 (tgt_CH→tgt) = N
     Total hops      = N
   At t = 40.0s, Transmission Success Rate = 0.XXXX
   ```

---

### Figure 2: Cluster Topology (Single Run)

**Script:** `C3-Evaluation of Cluster Reliability Single Simulation.py`

**What it does:**
- Builds the network and selects cluster heads
- Evaluates intra/inter-cluster reliability via exhaustive enumeration
- Visualizes the cluster topology with communication ranges
- Computes average max reliability for all intra/inter-cluster pairs

**Run command:**
```bash
python "C3-Evaluation of Cluster Reliability Based on Monte Carlo Simulation.py"
```

**Expected outputs:**
1. **Figure window 1:** C-MANET topology with:
   - Red triangles: Cluster heads
   - Light blue circles: Member nodes
   - Communication range circles (red for CHs, blue for members)
   - Title showing counts of intra/inter-cluster tasks

2. **Console output:**
   ```
   [Exhaustive Evaluation by MAX P_success over [0, 40.0]s]
   ✅ Intra-cluster Tasks: N pairs → Average Max Reliability = 0.XXXX
   ✅ Inter-cluster Tasks: N pairs → Average Max Reliability = 0.XXXX

   ✅ Exhaustive max-reliability evaluation completed.
   ```

---

### Figure 3: Monte Carlo Cluster Reliability Statistics

**Script:** `C2-Evaluation of Cluster Reliability in C-MANET.py`

**What it does:**
- Runs 500 Monte Carlo simulations
- Performs exhaustive evaluation of intra/inter-cluster reliability for each run
- Computes statistical measures (mean, std, variance)
- Visualizes reliability convergence and distributions

**Run command:**
```bash
python "C2-Evaluation of Cluster Reliability in C-MANET.py"
```

**Expected outputs:**
1. **Figure window 1:** Reliability vs. simulation run index with:
   - Red line: Intra-cluster reliability per run
   - Blue line: Inter-cluster reliability per run
   - Dashed reference lines: Mean values

2. **Figure window 2:** Histograms showing:
   - Left: Distribution of intra-cluster reliability
   - Right: Distribution of inter-cluster reliability

3. **Console output:**
   ```
   🚀 Starting Monte Carlo Simulation (500 runs)...

   [50/500] Done | Intra: 0.XXXX, Inter: 0.XXXX | Elapsed: XX.Xs
   ...
   [500/500] Done | Intra: 0.XXXX, Inter: 0.XXXX | Elapsed: XXX.Xs

   ============================================================
   ✅ Monte Carlo Simulation Results (500 runs):
   🔹 Intra-cluster Reliability:
       Mean = 0.XXXX, Std = 0.XXXX, Variance = 0.XXXXXX
   🔹 Inter-cluster Reliability:
       Mean = 0.XXXX, Std = 0.XXXX, Variance = 0.XXXXXX
   🔹 Ratio (Mean Intra / Mean Inter) = X.XX
   ============================================================

   ✅ Monte Carlo simulation and visualization completed.
   ```

---

### Experiment Results (E1, E2, E3)

**Scripts:** `E1-Correlated-mobility experiments.py`, `E2-interference-sensitivity experiment.py`, `E3-comparative experiment.py`

**What they do:**
- Run parameter sweeps across different R_ch_node values (50-95m)
- Save results to CSV files for post-processing
- Generate comparative analyses

**Run commands:**
```bash
python "E1-Correlated-mobility experiments.py"
python "E2-interference-sensitivity experiment.py"
python "E3-comparative experiment.py"
```

**Expected outputs:**
- **CSV files:** Results saved to `results/` directory
- **Console output:** Summary tables showing reliability metrics across parameter values

---

## Key Parameters Reference

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Area X-dimension | max_x | 100 m | Simulation area width |
| Area Y-dimension | max_y | 100 m | Simulation area height |
| Number of nodes | num_nodes | 50 | Total mobile nodes |
| Cluster head ratio | P | 0.12 (0.28 for E3) | Proportion of cluster heads |
| Mobility speed | v | 2.0 m/s | Node movement speed |
| Member comm. range | R_member | 30 m | Ordinary node range |
| CH comm. range | R_ch_node | 50-95 m | Cluster head range (variable in experiments) |
| Time slot duration | τ (tau) | 1.0 s | Transmission slot length |
| Max simulation time | T_MAX | 40.0 s | Maximum time for evaluation |
| Interference decay | β_I | 0.7 | Interference attenuation factor |
| Correlation strength | ρ | 0.8 | Group mobility correlation (E1) |
| Monte Carlo runs | NUM_MC | 200-500 | Number of MC iterations |

---

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'numpy' or 'pandas'**
   ```bash
   pip install numpy matplotlib pandas scipy
   ```

2. **Figures not displaying** (headless environment)
   ```python
   import matplotlib
   matplotlib.use('Agg')  # Set before importing pyplot
   ```

3. **Different results on each run**
   - Add random seed configuration (see [Random Seed Configuration](#random-seed-configuration))

4. **"No valid path found" message**
   - This is normal for some random topologies; re-run the script for a different network instance

5. **CSV files not being created**
   - Ensure the `results/` directory exists or is created by the script
   - Check write permissions in the current directory
