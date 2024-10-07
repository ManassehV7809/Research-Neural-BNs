import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns
import re  # Import regular expressions

def generate_architecture_impact_plots(csv_files_pattern):
    # Get the list of CSV files matching the pattern
    csv_files = glob.glob(csv_files_pattern)
    
    # Initialize a list to store all data
    all_data = []
    
    # Read and concatenate all CSV files
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Extract structure name and experiment number from the filename
        base_name = os.path.basename(csv_file)
        
        # Use regular expression to extract structure name and experiment number
        pattern = r'^(.*)_experiment_(\d+)_results\.csv$'
        match = re.match(pattern, base_name)
        if match:
            structure_name = match.group(1)
            experiment_num = match.group(2)
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
    
    # Ensure that the 'Structure', 'Node', and 'Experiment' columns are strings
    combined_df['Structure'] = combined_df['Structure'].astype(str)
    combined_df['Node'] = combined_df['Node'].astype(str)
    combined_df['Experiment'] = combined_df['Experiment'].astype(float)
    
    # Print unique 'Experiment' values
    print("Unique 'Experiment' values:", combined_df['Experiment'].unique())
    
    # Map 'Experiment' to the number of layers
    experiment_to_layers = {1.0: '2 Layers', 2.0: '4 Layers', 3.0: '6 Layers'}
    combined_df['Architecture'] = combined_df['Experiment'].map(experiment_to_layers)
    
    # Print unique 'Architecture' values
    print("Unique 'Architecture' values:", combined_df['Architecture'].unique())
    
    # Check for NaN values in 'KL Divergence NeuralBN'
    print("Any NaN in 'KL Divergence NeuralBN'?", combined_df['KL Divergence NeuralBN'].isnull().any())
    
    # For each structure, generate the plot
    for structure_name in combined_df['Structure'].unique():
        structure_df = combined_df[combined_df['Structure'] == structure_name]
        print(f"Processing Structure: {structure_name}")
        
        # Group by Architecture and Dataset Size, compute average KL Divergence
        avg_kl_df = structure_df.groupby(['Architecture', 'Dataset Size']).agg({
            'KL Divergence NeuralBN': 'mean',
        }).reset_index()
        
        # Verify that 'avg_kl_df' has data
        print("Grouped DataFrame head:")
        print(avg_kl_df.head())
        
        # Plotting
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=avg_kl_df,
            x='Dataset Size',
            y='KL Divergence NeuralBN',
            hue='Architecture',
            marker='o'
        )
        plt.xlabel('Dataset Size')
        plt.ylabel('Average KL Divergence (NeuralBN)')
        plt.title(f'Impact of Neural Network Architecture on KL Divergence\nStructure: {structure_name}')
        plt.legend(title='Architecture')
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"nn_architecture_impact_{structure_name}.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"Plot saved to {plot_filename}")
    
csv_files_pattern = '*_experiment_*_results.csv'
generate_architecture_impact_plots(csv_files_pattern)
