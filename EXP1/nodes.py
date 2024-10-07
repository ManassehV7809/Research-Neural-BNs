import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Define the structures and their edges
structures = {
    "chain_structure": [("A", "B"), ("B", "C")],
    "complex_structure": [("A", "B"), ("B", "C"), ("A", "C")],
    "ritesh_s_structure": [
        ("A", "D"),
        ("A", "F"),
        ("B", "D"),
        ("C", "D"),
        ("C", "E"),
        ("E", "G"),
        ("D", "G"),
        ("D", "F"),
        ("F", "H"),
        ("G", "H"),
        ("H", "I"),
        ("F", "I"),
    ],
    "vusani_s_structure_1": [
        ("A", "E"),
        ("B", "E"),
        ("C", "E"),
        ("D", "E"),
        ("E", "F"),
        ("E", "G"),
        ("E", "H"),
        ("F", "H"),
        ("G", "H"),
    ],
    "vusani_s_structure_2": [
        ("A", "F"),
        ("B", "F"),
        ("C", "F"),
        ("D", "F"),
        ("F", "G"),
        ("F", "H"),
        ("F", "I"),
        ("F", "J"),
        ("G", "J"),
        ("H", "J"),
        ("I", "J"),
    ],
    "complex_20_node_structure": [
        ('A', 'D'),
        ('A', 'E'),
        ('A', 'F'),
        ('B', 'D'),
        ('B', 'G'),
        ('B', 'H'),
        ('C', 'E'),
        ('C', 'G'),
        ('C', 'I'),
        ('D', 'J'),
        ('D', 'K'),
        ('E', 'J'),
        ('E', 'L'),
        ('F', 'K'),
        ('F', 'M'),
        ('G', 'L'),
        ('G', 'N'),
        ('H', 'M'),
        ('H', 'O'),
        ('I', 'N'),
        ('I', 'P'),
        ('J', 'Q'),
        ('K', 'Q'),
        ('L', 'R'),
        ('M', 'R'),
        ('N', 'S'),
        ('O', 'S'),
        ('P', 'T'),
        ('Q', 'T'),
        ('R', 'T'),
        ('S', 'T'),
    ],
}

def get_nodes_with_multiple_parents(structure_edges):
    """
    Returns a set of nodes that have multiple parents in the given structure.
    """
    from collections import defaultdict

    child_parents = defaultdict(set)
    for parent, child in structure_edges:
        child_parents[child].add(parent)

    nodes_with_multiple_parents = {node for node, parents in child_parents.items() if len(parents) > 1}
    return nodes_with_multiple_parents

def analyze_node_performance(csv_files_pattern):
    """
    Reads the experimental results from multiple CSV files and compares the performance of NeuralBN and TraditionalBN
    on nodes with multiple parents across all experiments and structures.
    """
    # Get the list of CSV files matching the pattern
    csv_files = glob.glob(csv_files_pattern)

    # Initialize a list to store all data
    all_data = []

    # Read and concatenate all CSV files
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Extract structure name and experiment number from the filename
        base_name = os.path.basename(csv_file)
        # Assuming filenames are like 'structure_name_experiment_X_results.csv'
        parts = base_name.split('_')
        if len(parts) >= 4 and parts[-1] == 'results.csv':
            structure_name = '_'.join(parts[:-3])  # Reconstruct the structure name
            experiment_num = parts[-3]
            df['Structure'] = structure_name
            df['Experiment'] = experiment_num
            all_data.append(df)
        else:
            print(f"Filename '{csv_file}' does not match expected pattern. Skipping.")
    
    # Combine all data into a single DataFrame
    if not all_data:
        print("No data found. Please check the CSV files pattern and filenames.")
        return
    combined_df = pd.concat(all_data, ignore_index=True)

    # Ensure that the 'Structure' and 'Node' columns are strings
    combined_df['Structure'] = combined_df['Structure'].astype(str)
    combined_df['Node'] = combined_df['Node'].astype(str)

    # Initialize a list to store results
    results = []

    # Iterate over each structure
    for structure_name in combined_df['Structure'].unique():
        structure_df = combined_df[combined_df['Structure'] == structure_name]
        print(f"\nAnalyzing Structure: {structure_name}")

        # Get the edges for the current structure
        structure_edges = structures.get(structure_name)
        if structure_edges is None:
            print(f"Structure '{structure_name}' not found in predefined structures.")
            continue

        # Identify nodes with multiple parents
        nodes_with_multiple_parents = get_nodes_with_multiple_parents(structure_edges)
        if not nodes_with_multiple_parents:
            print(f"No nodes with multiple parents in structure '{structure_name}'. Skipping.")
            continue
        print(f"Nodes with multiple parents: {nodes_with_multiple_parents}")

        # Filter the DataFrame to include only the nodes with multiple parents
        multi_parent_df = structure_df[structure_df['Node'].isin(nodes_with_multiple_parents)]

        # Group by Node, Experiment, and Dataset Size, then calculate average KL divergences
        node_performance = multi_parent_df.groupby(['Node', 'Experiment', 'Dataset Size']).agg({
            'KL Divergence NeuralBN': 'mean',
            'KL Divergence TraditionalBN': 'mean',
        }).reset_index()

        # Add structure information
        node_performance['Structure'] = structure_name

        # Append to results
        results.append(node_performance)

    # Combine all results into a single DataFrame
    results_df = pd.concat(results, ignore_index=True)

    # Display the results
    print("\nAverage KL Divergence for Nodes with Multiple Parents:")
    print(results_df)

    # Optionally, plot the results
    plot_node_performance(results_df)

def plot_node_performance(results_df):
    """
    Plots the average KL divergence for NeuralBN and TraditionalBN on nodes with multiple parents.
    Saves the plots to files instead of showing them.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Set up the plot style
    sns.set(style="whitegrid")

    # Create a directory to save the plots
    plots_dir = "node_performance_plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Iterate over each structure and plot the performance
    for structure_name in results_df['Structure'].unique():
        structure_df = results_df[results_df['Structure'] == structure_name]
        nodes = structure_df['Node'].unique()

        for node in nodes:
            node_df = structure_df[structure_df['Node'] == node]

            # Melt the DataFrame to long format
            node_melted_df = pd.melt(
                node_df,
                id_vars=['Dataset Size', 'Experiment'],
                value_vars=['KL Divergence NeuralBN', 'KL Divergence TraditionalBN'],
                var_name='Model',
                value_name='KL Divergence'
            )

            # Map the 'Model' values to more readable labels
            node_melted_df['Model'] = node_melted_df['Model'].replace({
                'KL Divergence NeuralBN': 'NeuralBN',
                'KL Divergence TraditionalBN': 'TraditionalBN'
            })

            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=node_melted_df,
                x='Dataset Size',
                y='KL Divergence',
                hue='Model',
                style='Experiment',
                markers=True,
                dashes=False
            )
            plt.xlabel('Dataset Size')
            plt.ylabel('Average KL Divergence')
            plt.title(f'Average KL Divergence for Node {node} in {structure_name}')
            plt.legend()
            plt.tight_layout()

            # Save the plot to a file
            filename = f"{structure_name}_node_{node}_performance.png"
            filepath = os.path.join(plots_dir, filename)
            plt.savefig(filepath)
            plt.close()  # Close the figure to free memory

            print(f"Plot saved to {filepath}")

# Example usage:
# Assuming your CSV files are named like 'structure_name_experiment_X_results.csv'
# and are located in the current directory or specify the correct path
csv_files_pattern = '*_experiment_*_results.csv'
analyze_node_performance(csv_files_pattern)
