import pandas as pd
import os

# List of structures
structures = ["chain_structure", "triangular_structure", "tree_structure",
              "sparse_structure", "sparse_structure_2", "dense_structure"]

# Number of experiments (assuming 3 as per your initial script)
n_experiments = 3

# Directory containing raw CSV files
raw_data_dir = "experiment results"  # Change if your raw CSVs are in a different directory

# Directory to save aggregated CSV files
aggregated_data_dir = "experiment_results"
os.makedirs(aggregated_data_dir, exist_ok=True)

for structure in structures:
    exps=['1.0','2.0','3.0']
    for exp_num in exps:
        # Construct raw CSV filename
        raw_filename = f"{structure}_experiment_{exp_num}_results.csv"
        raw_filepath = os.path.join(raw_data_dir, raw_filename)
        
        if not os.path.exists(raw_filepath):
            print(f"File {raw_filepath} not found. Skipping.")
            continue
        
        # Read raw CSV
        df = pd.read_csv(raw_filepath)
        
        # Group by 'Dataset Size' and compute mean and std
        grouped = df.groupby("Dataset Size").agg({
            "KL Divergence NeuralBN": ['mean', 'std'],
            "KL Divergence TraditionalBN": ['mean', 'std']
        }).reset_index()
        
        # Rename columns for LaTeX compatibility
        grouped.columns = ['Dataset Size', 
                           'KL Divergence NeuralBN_mean', 'KL Divergence NeuralBN_std',
                           'KL Divergence TraditionalBN_mean', 'KL Divergence TraditionalBN_std']
        
        # Save aggregated CSV
        aggregated_filename = f"{structure}_experiment_{exp_num}_results.csv"
        aggregated_filepath = os.path.join(aggregated_data_dir, aggregated_filename)
        grouped.to_csv(aggregated_filepath, index=False)
        
        print(f"Aggregated data saved to {aggregated_filepath}")
