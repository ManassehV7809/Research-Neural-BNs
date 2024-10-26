import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# Path to the folder containing CSV files
folder_path = './kfoldexps/'  

# Structures with their corresponding edges
structures = {
    "chain_structure": [("A", "B"), ("B", "C")],
    "triangular_structure": [("A", "B"), ("B", "C"), ("A", "C")],
    "tree_structure": [
        ("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F"), ("C", "G"),
        ("D", "H"), ("E", "I"), ("F", "J"), ("G", "K")
    ],
    "sparse_structure": [
        ("A", "E"), ("B", "E"), ("C", "E"), ("D", "F"), ("E", "F"), ("E", "G"),
        ("F", "H"), ("G", "H")
    ],
    "sparse_structure_2": [
        ("A", "G"), ("B", "G"), ("C", "G"), ("D", "H"), ("G", "I"), ("H", "I"),
        ("E", "H"), ("I", "J"), ("H", "J"), ("F", "J")
    ],
    "dense_structure": [
        ('A', 'D'), ('A', 'E'), ('A', 'F'), ('B', 'D'), ('B', 'G'), ('B', 'H'), 
        ('C', 'E'), ('C', 'G'), ('C', 'I'), ('D', 'J'), ('D', 'K'), ('E', 'J'), 
        ('E', 'L'), ('F', 'K'), ('F', 'M'), ('G', 'L'), ('G', 'N'), ('H', 'M'), 
        ('H', 'O'), ('I', 'N'), ('I', 'P'), ('J', 'Q'), ('K', 'Q'), ('L', 'R'), 
        ('M', 'R'), ('N', 'S'), ('O', 'S'), ('P', 'T'), ('Q', 'T'), ('R', 'T'), 
        ('Q', 'U'), ('R', 'U'), ('S', 'U')
    ]
}

# Function to calculate the indegree of nodes
def calculate_indegree(edges):
    indegree_count = defaultdict(int)
    for parent, child in edges:
        indegree_count[child] += 1
    return indegree_count

# List of experiments
experiments = ['experiment_1', 'experiment_2', 'experiment_3']

# Iterate through each structure and experiment to load data and generate plots for nodes with indegree of 3
for structure_name, edges in structures.items():
    # Calculate the indegree for each node in the structure
    indegree_count = calculate_indegree(edges)
    
    # Find nodes with an indegree of 3
    nodes_with_indegree_3 = [node for node, indegree in indegree_count.items() if indegree == 3]
    
    # Skip if no nodes with indegree 3
    if not nodes_with_indegree_3:
        print(f"No nodes with indegree 3 in {structure_name}")
        continue

    for experiment in experiments:
        # Construct the filename based on structure and experiment
        file_name = f"{structure_name}_{experiment}_results.csv"
        file_path = os.path.join(folder_path, file_name)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Filter for nodes that are in the list of nodes with indegree of 3
        indegree_3_nodes = df[df['Node'].isin(nodes_with_indegree_3)]
        
        if indegree_3_nodes.empty:
            print(f"No nodes with indegree of 3 found in {file_name}")
            continue
        
        # Generate the plot for nodes with indegree of 3
        plt.figure(figsize=(10, 6))
        
        sns.lineplot(x='Dataset Size', y='KL Divergence NeuralBN', data=indegree_3_nodes, label='NeuralBN KL Divergence', marker='o', color='green')
        sns.lineplot(x='Dataset Size', y='KL Divergence TraditionalBN', data=indegree_3_nodes, label='TraditionalBN KL Divergence', marker='o', color='red')
        
        # Customize the plot
        plt.title(f'KL Divergence for Nodes with Indegree=3: {structure_name.replace("_", " ").capitalize()} ({experiment.replace("_", " ").capitalize()})')
        plt.xlabel('Dataset Size')
        plt.ylabel('KL Divergence')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        output_filename = f"{structure_name}_{experiment}_indegree_3_plot.png"
        output_path = os.path.join(folder_path, output_filename)
        plt.savefig(output_path)
        plt.close()

        print(f"Plot for indegree=3 nodes saved: {output_path}")
