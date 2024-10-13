import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the path where the results CSV files are stored
results_path = "./"

# Define the list of structures used in the experiments
structures = [
    "chain_structure",
    "complex_structure",
    "ritesh_s_structure",
    "vusani_s_structure_1",
    "vusani_s_structure_2",
    "complex_20_node_structure",
]

# Create empty lists to store the average KL divergence for each structure
neural_kl_divergences = []
traditional_kl_divergences = []

# Iterate over each structure and load the corresponding results
for structure in structures:
    # Construct the file name for the results of the second experiment
    file_name = f"{structure}_experiment_1.0_results.csv"
    file_path = os.path.join(results_path, file_name)

    # Load the CSV file into a DataFrame
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        # Calculate the average KL divergence for NeuralBN and TraditionalBN for the given structure
        avg_kl_neural = df["KL Divergence NeuralBN"].mean()
        avg_kl_traditional = df["KL Divergence TraditionalBN"].mean()

        # Append the average KL divergence values to the respective lists
        neural_kl_divergences.append(avg_kl_neural)
        traditional_kl_divergences.append(avg_kl_traditional)
    else:
        print(f"Results file for {structure} not found.")

# Plotting the results as a bar graph
x_labels = [
    "Chain Structure",
    "Complex Structure",
    "Ritesh's Structure",
    "Vusani's Structure 1",
    "Vusani's Structure 2",
    "Complex 20-Node Structure",
]
x = range(len(structures))

# Set the width of the bars
bar_width = 0.35

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the bars for NeuralBN and TraditionalBN
bars1 = ax.bar([i - bar_width / 2 for i in x], neural_kl_divergences, bar_width, label='NeuralBN', color='b')
bars2 = ax.bar([i + bar_width / 2 for i in x], traditional_kl_divergences, bar_width, label='TraditionalBN', color='g')

# Set the labels, title, and legend
ax.set_xlabel('Network Structure')
ax.set_ylabel('Average KL Divergence')
ax.set_title('Average KL Divergence vs. Network Structure (NeuralBN vs. TraditionalBN)')
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=45, ha='right')
ax.legend()

# Adjust layout for better fit
plt.tight_layout()

# Show the plot
plt.show()