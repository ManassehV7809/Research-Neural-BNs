import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to the folder containing CSV files
folder_path = './kfoldexps/'  # Change this to the actual path

# List of structures and experiments
structures = ['chain_structure', 'dense_structure', 'sparse_structure', 'sparse_structure_2', 'tree_structure', 'triangular_structure']
experiments = ['experiment_1', 'experiment_2', 'experiment_3']

# Iterate through each structure and experiment to load data and generate plots
for structure in structures:
    for experiment in experiments:
        # Construct the filename based on structure and experiment
        file_name = f"{structure}_{experiment}_results.csv"
        file_path = os.path.join(folder_path, file_name)
        
        # Check if the file exists
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            continue
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Ensure that 'Dataset Size' is of integer type
        df['Dataset Size'] = df['Dataset Size'].astype(int)
        
        # Group by 'Dataset Size' and compute mean and std for both NeuralBN and TraditionalBN
        agg_df = df.groupby('Dataset Size').agg({
            'KL Divergence NeuralBN': ['mean', 'std'],
            'KL Divergence TraditionalBN': ['mean', 'std']
        }).reset_index()
        
        # Flatten the MultiIndex columns
        agg_df.columns = ['Dataset Size', 'KL_NeuralBN_mean', 'KL_NeuralBN_std', 'KL_TraditionalBN_mean', 'KL_TraditionalBN_std']
        
        # Generate the plot
        plt.figure(figsize=(10, 6))
        
        # Plot for NeuralBN
        plt.errorbar(agg_df['Dataset Size'], agg_df['KL_NeuralBN_mean'], yerr=agg_df['KL_NeuralBN_std'],
                     fmt='-o', capsize=5, label='NeuralBN', color='green', ecolor='lightgreen')
        
        # Plot for TraditionalBN
        plt.errorbar(agg_df['Dataset Size'], agg_df['KL_TraditionalBN_mean'], yerr=agg_df['KL_TraditionalBN_std'],
                     fmt='-s', capsize=5, label='TraditionalBN', color='red', ecolor='lightcoral')
        
        # Set y-axis limits to zoom in
        plt.ylim(0, 1.5)  # Set the y-axis from 0 to 2
        
        # Customize the plot
        plt.title(f'KL Divergence Comparison: {structure.replace("_", " ").title()} ({experiment.replace("_", " ").title()})')
        plt.xlabel('Dataset Size')
        plt.ylabel('KL Divergence')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        output_filename = f"{structure}_{experiment}_plot.png"
        output_path = os.path.join(folder_path, output_filename)
        plt.savefig(output_path)
        plt.close()
        
        print(f"Plot saved: {output_path}")
