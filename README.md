# README

This project contains two directories, **EXP1** and **EXP2**, which include everything needed to run two separate experiments. Follow the instructions below to run each experiment and analyze the results.

---

### Directory Structure

- **EXP1** - Contains scripts and files needed for the first experiment.
- **EXP2** - Contains scripts and files needed for the second experiment.

---

## EXP1

In the **EXP1** directory, you can perform the following steps to run and analyze the first experiment:

1. **Run the Main Experiment:**
   - Execute `model_2.py` to start the main experiment. This script will produce multiple CSV files containing results from all folds, all iterations, across all 3 experiments.

2. **Analysis and Post-processing:**
   - **KL Divergence Analysis**: Run `nodes.py` to check the KL divergence on nodes with an indegree of 3.
   - **Impact Analysis**: Run `impact.py` to analyze the impact of neural network architecture on the results.
   - **Aggregation**: Run `agg.py` to aggregate the results in a structured format.
   - **Plotting**: Run `plot.py` to visualize the aggregated results.

---

## EXP2

In the **EXP2** directory, the following steps will guide you through the second experiment:

1. **Run the Main Experiment:**
   - Execute `model.py` to start the main experiment. This script will produce multiple CSV files containing results from all folds, all iterations, across all 3 experiments for the specified dataset size.

2. **Plotting**:
   - Run `plot.py` to visualize the results.

