import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# List of structures
structures = ["chain_structure", "complex_structure", "ritesh_s_structure","vusani_s_structure_1","vusani_s_structure_2","complex_20_node_structure"]

# Colors for NeuralBN and TraditionalBN
neural_colors = ["blue", "orange", "purple"]  # Colors for NeuralBN for each experiment
traditional_colors = ["green", "red", "brown"]  # Colors for TraditionalBN for each experiment

# Iterate over each structure
for structure_name in structures:
    # Iterate over each experiment (architecture)
    for experiment_num in range(3):
        plt.figure(figsize=(20, 12))
        
        # Read the results CSV file for this experiment
        filename = f"{structure_name}_experiment_{experiment_num + 1}_results.csv"
        if not os.path.exists(filename):
            print(f"File {filename} not found. Skipping.")
            continue

        df = pd.read_csv(filename)

        # Group the data by Dataset Size and compute mean and std deviation
        grouped = df.groupby("Dataset Size").agg({
            "KL Divergence NeuralBN": ['mean', 'std'],
            "KL Divergence TraditionalBN": ['mean', 'std']
        }).reset_index()

        # Extract means and std deviations
        dataset_sizes = grouped["Dataset Size"]
        neural_mean = grouped[("KL Divergence NeuralBN", "mean")]
        neural_std = grouped[("KL Divergence NeuralBN", "std")]
        traditional_mean = grouped[("KL Divergence TraditionalBN", "mean")]
        traditional_std = grouped[("KL Divergence TraditionalBN", "std")]

        # Plotting NeuralBN
        plt.errorbar(
            dataset_sizes,
            neural_mean,
            yerr=neural_std,
            label=f"NeuralBN",
            fmt='-o',
            color=neural_colors[experiment_num % len(neural_colors)],
            capsize=5,
        )
        # Plotting TraditionalBN
        plt.errorbar(
            dataset_sizes,
            traditional_mean,
            yerr=traditional_std,
            label=f"TraditionalBN",
            fmt='--s',
            color=traditional_colors[experiment_num % len(traditional_colors)],
            capsize=5,
        )

        plt.title(f"KL Divergence vs Dataset Size for {structure_name} Experiment {experiment_num + 1}")
        plt.xlabel("Dataset Size")
        plt.ylabel("KL Divergence")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plt.savefig(f"{structure_name}_experiment_{experiment_num + 1}_kl_divergence_plot.png")
        plt.close()
        print(f"Plot saved for {structure_name} Experiment {experiment_num + 1}")
