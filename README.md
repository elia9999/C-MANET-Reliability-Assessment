# C-MANET Reliability Assessment

Python simulations for evaluating cluster reliability in Cooperative MANETs (C-MANET).

---

## Recommended Python version

- Python 3.8 — 3.11 (64-bit recommended)

## Key dependencies

- numpy >= 1.20
- matplotlib >= 3.3



## Files / Modules and purpose

- `Simulation of a Single Communication Task in C-MANET.py` — This script simulates a single source-to-target communication task, computes time-varying path stability and transmission success, and visualizes the path and plots.
- `Evaluation of Cluster Reliability Based on Monte Carlo Simulation.py` — This script builds the network, selects cluster heads, visualizes the topology, and performs an exhaustive evaluation of intra-cluster and inter-cluster reliability.
- `Non-Independent Case.py` — This script provides a Monte Carlo study that compares independent versus correlated (group) mobility and computes intra-cluster and inter-cluster reliability.
- `Interference.py` — This script conducts a Monte Carlo sensitivity study on interference and evaluates intra-cluster and inter-cluster reliability under different interference levels.

Common utilities used across scripts:

- `numpy` — numeric operations, time grids, averaging
- `matplotlib` — plotting and visualization
- `collections.deque` — BFS queues
- `math`, `random` — geometry and randomness

