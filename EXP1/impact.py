import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to the folder containing CSV files
folder_path = './kfoldexps/'  # Change this to the actual path

# Structures with their corresponding CSV data
structures = ['chain_structure', 'triangular_structure', 'tree_structure', 'sparse_structure', 'sparse_structure_2', 'dense_structure']

# Map experiments to their respective neural network depths
experiment_depth_map = {
    'experiment_1': '2 layers',
    'experiment_2': '4 layers',
    'experiment_3': '6 layers'
}

# Iterate through each structure and experiment to load data and generate architecture impact plots
for structure in structures:
    plt.figure(figsize=(10, 6))
    
    # Plot for each experiment (which corresponds to different neural network depths)
    for experiment in experiment_depth_map:
        # Construct the filename based on structure and experiment
        file_name = f"{structure}_{experiment}_results.csv"
        file_path = os.path.join(folder_path, file_name)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Plot KL Divergence vs Dataset Size for the given depth
        sns.lineplot(x='Dataset Size', y='KL Divergence NeuralBN', data=df, label=f'{experiment_depth_map[experiment]}', marker='o')
    
    # Customize the plot
    plt.title(f'Impact of Neural Network Architecture on KL Divergence for {structure.replace("_", " ").capitalize()}')
    plt.xlabel('Dataset Size')
    plt.ylabel('KL Divergence')
    plt.legend(title='Neural Network Depth')
    plt.grid(True)
    
    # Save the plot
    output_filename = f"{structure}_nn_architecture_impact.png"
    output_path = os.path.join(folder_path, output_filename)
    plt.savefig(output_path)
    plt.close()

    print(f"Architecture impact plot saved: {output_path}")
