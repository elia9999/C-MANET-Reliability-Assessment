# C-MANET Reliability Assessment

Python simulations for evaluating cluster reliability in Cooperative MANETs (C-MANET).

---

## Recommended Python version

- Python 3.8 — 3.11 (64-bit recommended)

## Key dependencies

- numpy >= 1.20
- matplotlib >= 3.3



## Files / Modules and purpose

- `Simulation of a Single Communication Task in C-MANET.py` — simulates a single source→target communication task, computes time-varying path stability and transmission success, and visualizes the path and plots.
- `Evaluation of Cluster Reliability Based on Monte Carlo Simulation.py` — builds network, selects cluster heads, visualizes topology, and performs exhaustive max-P_success evaluation for intra/inter-cluster reliability.
- `Non-Independent Case.py` — Monte Carlo study comparing independent vs. correlated (group) mobility; computes intra/inter reliability.
- `Interference.py` — Monte Carlo sensitivity study on interference; evaluates intra/inter cluster reliability under different interference levels.

Common utilities used across scripts:

- `numpy` — numeric operations, time grids, averaging
- `matplotlib` — plotting and visualization
- `collections.deque` — BFS queues
- `math`, `random` — geometry and randomness

