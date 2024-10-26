import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the directory containing the CSV files
data_dir = '.'  # Change if your CSVs are in a different directory

# Define the CSV filenames for all experiments
csv_files = [
    'chain_structure_experiment_1_results.csv',
    'triangular_structure_experiment_1_results.csv',
    'tree_structure_experiment_1_results.csv',
    'sparse_structure_experiment_1_results.csv',
    'sparse_structure_2_experiment_1_results.csv',
    'dense_structure_experiment_1_results.csv',
    'chain_structure_experiment_2_results.csv',
    'triangular_structure_experiment_2_results.csv',
    'tree_structure_experiment_2_results.csv',
    'sparse_structure_experiment_2_results.csv',
    'sparse_structure_2_experiment_2_results.csv',
    'dense_structure_experiment_2_results.csv',
    'chain_structure_experiment_3_results.csv',
    'triangular_structure_experiment_3_results.csv',
    'tree_structure_experiment_3_results.csv',
    'sparse_structure_experiment_3_results.csv',
    'sparse_structure_2_experiment_3_results.csv',
    'dense_structure_experiment_3_results.csv'
]

# Define the neural network architectures used in the experiments
architectures = {
    1: '2 Hidden Layers',
    2: '4 Hidden Layers',
    3: '6 Hidden Layers'
}

# Define a dictionary to hold the data for each structure and architecture
data = {arch: {} for arch in architectures.values()}

# Load the data from the CSV files
for csv_file in csv_files:
    structure_name = csv_file.split('_experiment_')[0]
    df = pd.read_csv(os.path.join(data_dir, csv_file))
    
    # Extract data for each experiment (1, 2, 3)
    for experiment, architecture in architectures.items():
        arch_data = df[df['Experiment'] == experiment]
        avg_kl_divergence_neural = arch_data['KL Divergence NeuralBN'].mean()
        avg_kl_divergence_traditional = arch_data['KL Divergence TraditionalBN'].mean()
        if structure_name not in data[architecture]:
            data[architecture][structure_name] = {
                'NeuralBN': avg_kl_divergence_neural,
                'TraditionalBN': avg_kl_divergence_traditional
            }
        else:
            data[architecture][structure_name]['NeuralBN'] = avg_kl_divergence_neural
            data[architecture][structure_name]['TraditionalBN'] = avg_kl_divergence_traditional

# Plot the bar graphs for each architecture
for experiment, architecture in architectures.items():
    plt.figure(figsize=(12, 8))
    structures = list(data[architecture].keys())
    neural_kl = [data[architecture][structure]['NeuralBN'] for structure in structures]
    traditional_kl = [data[architecture][structure]['TraditionalBN'] for structure in structures]
    
    x = range(len(structures))
    width = 0.35
    
    plt.bar(x, neural_kl, width, label='NeuralBN', color='skyblue')
    plt.bar([i + width for i in x], traditional_kl, width, label='TraditionalBN', color='lightcoral')
    
    plt.xlabel('Network Structure')
    plt.ylabel('Average KL Divergence')
    plt.title(f'Experiment {experiment}: KL Divergence Across Structures ({architecture})')
    plt.xticks([i + width / 2 for i in x], structures, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'kl_divergence_experiment_{experiment}_{architecture.replace(" ", "_").lower()}.png')
    plt.show()
