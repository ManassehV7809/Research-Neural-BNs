import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
import matplotlib.pyplot as plt
from pgmpy.estimators import MaximumLikelihoodEstimator
import time

# Determine the device being used for PyTorch
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device for PyTorch")

# Suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class CPDNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(CPDNetwork, self).__init__()
        # Randomly decide on the number of layers and units per layer
        layers_config = [np.random.choice([32,64,128, 256, 512]) for _ in range(np.random.choice([2,3]))]
        layers = []
        for units in layers_config:
            layers.append(nn.Linear(input_size, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            layers.append(nn.BatchNorm1d(units))
            input_size = units
        layers.append(nn.Linear(input_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.network:
            if isinstance(layer, nn.BatchNorm1d):
                if x.size(0) == 1:  # If the batch size is 1, skip BatchNorm
                    continue
            x = layer(x)
        return x


class NeuralBayesianNetwork(BayesianNetwork):
    def __init__(self, ebunch=None):
        super(NeuralBayesianNetwork, self).__init__(ebunch)
        self.models = {}
        self.architecture_log = []

    def fit(self, data, epochs=100, batch_size=128, patience=30):
        for node in self.nodes():
            parents = list(self.get_parents(node))
            self._add_cpd_with_nn(node, parents, data, epochs, batch_size, patience)
        self.check_model()

    def _add_cpd_with_nn(self, variable, evidence, data, epochs, batch_size, patience):
        X = data[evidence].values.astype('float32') if evidence else np.zeros((data.shape[0], 1)).astype('float32')
        y = data[variable].values
        input_size = X.shape[1] if evidence else 1

        # Convert data to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(device)

        # Create and train the model with random architecture
        model, architecture_details = self._train_nn(X_tensor, y_tensor, input_size, 2, epochs, batch_size, patience)
        self.models[variable] = model
        self.architecture_log.append((variable, architecture_details))

        # Predict CPD values
        if evidence:
            evidence_values = [list(range(2)) for _ in evidence]  # Assuming binary variables
            evidence_combinations = np.array(np.meshgrid(*evidence_values)).T.reshape(-1, input_size)
        else:
            evidence_combinations = np.array([[0]])

        evidence_tensor = torch.tensor(evidence_combinations, dtype=torch.float32).to(device)
        with torch.no_grad():
            predicted_cpd = model(evidence_tensor).cpu().numpy()

        # Normalize the predicted CPD to ensure it sums to 1
        predicted_cpd /= predicted_cpd.sum(axis=1, keepdims=True)

        # Create and add TabularCPD
        if evidence:
            cpd = TabularCPD(variable=variable, variable_card=2, values=predicted_cpd.T,
                             evidence=evidence, evidence_card=[2] * len(evidence))
        else:
            cpd = TabularCPD(variable=variable, variable_card=2, values=predicted_cpd.T)

        self.add_cpds(cpd)

    def _train_nn(self, X_train, y_train, input_size, output_size, epochs, batch_size, patience):
        model = CPDNetwork(input_size, output_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  
        best_loss = float('inf')
        best_model_state = None
        early_stop_counter = 0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model_state = model.state_dict()
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                break

        model.load_state_dict(best_model_state)

        # Log the architecture details
        architecture_details = {
            'layers_config': model.network,
            'loss': best_loss,
        }

        return model, architecture_details

    def sample(self, n):
        sampler = BayesianModelSampling(self)
        samples = sampler.forward_sample(size=n)
        return samples

def generate_random_cpds(structure_name):
    if structure_name == 'chain_structure':
        prob_A = np.random.rand()
        cpd_A = TabularCPD(variable='A', variable_card=2, values=[[prob_A], [1 - prob_A]])

        prob_B = np.random.rand(2)
        prob_B = prob_B / prob_B.sum()
        cpd_B = TabularCPD(variable='B', variable_card=2, values=[prob_B, 1 - prob_B], evidence=['A'], evidence_card=[2])

        prob_C = np.random.rand(2)
        prob_C = prob_C / prob_C.sum()
        cpd_C = TabularCPD(variable='C', variable_card=2, values=[prob_C, 1 - prob_C], evidence=['B'], evidence_card=[2])

        return [cpd_A, cpd_B, cpd_C]

    elif structure_name == 'complex_structure':
        prob_A = np.random.rand()
        cpd_A = TabularCPD(variable='A', variable_card=2, values=[[prob_A], [1 - prob_A]])

        prob_B = np.random.rand(2)
        prob_B = prob_B / prob_B.sum()
        cpd_B = TabularCPD(variable='B', variable_card=2, values=[prob_B, 1 - prob_B], evidence=['A'], evidence_card=[2])

        prob_C = np.random.rand(4)
        prob_C = prob_C / prob_C.sum()
        cpd_C = TabularCPD(variable='C', variable_card=2, 
                           values=[prob_C, 1 - prob_C], 
                           evidence=['A', 'B'], evidence_card=[2, 2])

        return [cpd_A, cpd_B, cpd_C]

    elif structure_name == 'v_structure':
        prob_A = np.random.rand()
        cpd_A = TabularCPD(variable='A', variable_card=2, values=[[prob_A], [1 - prob_A]])

        prob_B = np.random.rand()
        cpd_B = TabularCPD(variable='B', variable_card=2, values=[[prob_B], [1 - prob_B]])

        prob_C = np.random.rand(4)
        prob_C = prob_C / prob_C.sum()
        cpd_C = TabularCPD(variable='C', variable_card=2,
                           values=[prob_C, 1 - prob_C],
                           evidence=['A', 'B'], evidence_card=[2, 2])

        return [cpd_A, cpd_B, cpd_C]

    elif structure_name == 'fully_connected':
        prob_A = np.random.rand()
        cpd_A = TabularCPD(variable='A', variable_card=2, values=[[prob_A], [1 - prob_A]])

        prob_B = np.random.rand(2)
        prob_B = prob_B / prob_B.sum()
        cpd_B = TabularCPD(variable='B', variable_card=2, 
                           values=[prob_B, 1 - prob_B], 
                           evidence=['A'], evidence_card=[2])

        prob_C = np.random.rand(4)
        prob_C = prob_C / prob_C.sum()
        cpd_C = TabularCPD(variable='C', variable_card=2,
                           values=[prob_C, 1 - prob_C],
                           evidence=['A', 'B'], evidence_card=[2, 2])

        return [cpd_A, cpd_B, cpd_C]

    elif structure_name == 'diamond_structure':
        prob_A = np.random.rand()
        cpd_A = TabularCPD(variable='A', variable_card=2, values=[[prob_A], [1 - prob_A]])

        prob_B = np.random.rand(2)
        prob_B = prob_B / prob_B.sum()
        cpd_B = TabularCPD(variable='B', variable_card=2, values=[prob_B, 1 - prob_B], evidence=['A'], evidence_card=[2])

        prob_C = np.random.rand(2)
        prob_C = prob_C / prob_C.sum()
        cpd_C = TabularCPD(variable='C', variable_card=2, values=[prob_C, 1 - prob_C], evidence=['A'], evidence_card=[2])

        prob_D = np.random.rand(4)
        prob_D = prob_D / prob_D.sum()
        cpd_D = TabularCPD(variable='D', variable_card=2,
                           values=[prob_D, 1 - prob_D],
                           evidence=['B', 'C'], evidence_card=[2, 2])

        return [cpd_A, cpd_B, cpd_C, cpd_D]

    elif structure_name == 'tree_structure':
        prob_A = np.random.rand()
        cpd_A = TabularCPD(variable='A', variable_card=2, values=[[prob_A], [1 - prob_A]])

        prob_B = np.random.rand(2)
        prob_B = prob_B / prob_B.sum()
        cpd_B = TabularCPD(variable='B', variable_card=2, values=[prob_B, 1 - prob_B], evidence=['A'], evidence_card=[2])

        prob_C = np.random.rand(2)
        prob_C = prob_C / prob_C.sum()
        cpd_C = TabularCPD(variable='C', variable_card=2, values=[prob_C, 1 - prob_C], evidence=['A'], evidence_card=[2])

        prob_D = np.random.rand(2)
        prob_D = prob_D / prob_D.sum()
        cpd_D = TabularCPD(variable='D', variable_card=2, values=[prob_D, 1 - prob_D], evidence=['B'], evidence_card=[2])

        prob_E = np.random.rand(2)
        prob_E = prob_E / prob_E.sum()
        cpd_E = TabularCPD(variable='E', variable_card=2, values=[prob_E, 1 - prob_E], evidence=['B'], evidence_card=[2])

        prob_F = np.random.rand(2)
        prob_F = prob_F / prob_F.sum()
        cpd_F = TabularCPD(variable='F', variable_card=2, values=[prob_F, 1 - prob_F], evidence=['C'], evidence_card=[2])

        return [cpd_A, cpd_B, cpd_C, cpd_D, cpd_E, cpd_F]

    elif structure_name == 'polytree_structure':
        prob_A = np.random.rand()
        cpd_A = TabularCPD(variable='A', variable_card=2, values=[[prob_A], [1 - prob_A]])

        prob_B = np.random.rand()
        cpd_B = TabularCPD(variable='B', variable_card=2, values=[[prob_B], [1 - prob_B]])

        prob_C = np.random.rand(4)
        prob_C = prob_C / prob_C.sum()
        cpd_C = TabularCPD(variable='C', variable_card=2,
                           values=[prob_C, 1 - prob_C],
                           evidence=['A', 'B'], evidence_card=[2, 2])

        prob_D = np.random.rand(2)
        prob_D = prob_D / prob_D.sum()
        cpd_D = TabularCPD(variable='D', variable_card=2, values=[prob_D, 1 - prob_D], evidence=['C'], evidence_card=[2])

        prob_E = np.random.rand(2)
        prob_E = prob_E / prob_E.sum()
        cpd_E = TabularCPD(variable='E', variable_card=2, values=[prob_E, 1 - prob_E], evidence=['C'], evidence_card=[2])

        prob_F = np.random.rand(4)
        prob_F = prob_F / prob_F.sum()
        cpd_F = TabularCPD(variable='F', variable_card=2,
                           values=[prob_F, 1 - prob_F],
                           evidence=['D', 'E'], evidence_card=[2, 2])

        return [cpd_A, cpd_B, cpd_C, cpd_D, cpd_E, cpd_F]

def generate_synthetic_data(structure, cpds, size):
    bn = BayesianNetwork(structure)
    for cpd in cpds:
        bn.add_cpds(cpd)
    bn.check_model()
    sampler = BayesianModelSampling(bn)
    data = sampler.forward_sample(size=size)
    return pd.DataFrame(data)

def create_traditional_bn(data_df, structure):
    traditional_bn = BayesianNetwork(structure)
    traditional_bn.fit(data_df, estimator=MaximumLikelihoodEstimator)
    # from pgmpy.estimators import BayesianEstimator
    # # Fit the model using BayesianEstimator
    # traditional_bn.fit(data_df, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=5)

    return traditional_bn
# Main execution
dataset_sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]  # Different dataset sizes for testing
iterations = 10  # Number of iterations for each setup
results = []
architecture_log_file = "neuralbn_architectures.txt"

model_structures = {
    'chain_structure': [('A', 'B'), ('B', 'C')],
    'complex_structure': [('A', 'B'), ('B', 'C'), ('A', 'C')],
    'v_structure': [('A', 'C'), ('B', 'C')],
}

with open(architecture_log_file, "w") as log_file:
    for structure_name, structure in model_structures.items():
        log_file.write(f"\nEvaluating structure: {structure_name}\n")
        print(f"\nEvaluating structure: {structure_name}")
        
        for dataset_size in dataset_sizes:
            for iteration in range(iterations):
                print(f"\nDataset Size: {dataset_size}, Iteration: {iteration + 1}")
                
                # Generate random CPDs for each iteration
                cpds = generate_random_cpds(structure_name)
                
                # Generate synthetic data
                data_df = generate_synthetic_data(structure, cpds, dataset_size)
                
                # Split the data into training and testing sets
                train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=iteration)
                
                # Train NeuralBN on the training set
                neural_bn = NeuralBayesianNetwork(structure)
                start_time = time.time()
                neural_bn.fit(train_df)
                neural_time = time.time() - start_time
                print(f"Training time for NeuralBN: {neural_time:.4f} seconds")

                # Log the architecture of the NeuralBN to the file
                log_file.write(f"NeuralBN Architecture for iteration {iteration + 1} (Dataset Size: {dataset_size}):\n")
                for log_entry in neural_bn.architecture_log:
                    variable, architecture = log_entry
                    log_file.write(f"Variable {variable}: Layers Config {architecture['layers_config']}, Loss: {architecture['loss']:.4f}\n")
                log_file.write("\n")

                # Train traditional BN on the training set
                start_time = time.time()
                traditional_bn = create_traditional_bn(data_df, structure)
                traditional_time = time.time() - start_time
                print(f"Training time for TraditionalBN: {traditional_time:.4f} seconds")

                # Define original_bn for comparison
                original_bn = BayesianNetwork(structure)
                for cpd in cpds:
                    original_bn.add_cpds(cpd)
                original_bn.check_model()
                
                def smooth_distribution(distribution, epsilon=1e-5):
                    # Apply smoothing by adding a small epsilon and normalize
                    distribution = np.clip(distribution + epsilon, a_min=epsilon, a_max=None)
                    return distribution / np.sum(distribution)


                def compare_with_ground_truth(variable):
                    original_cpd = original_bn.get_cpds(variable).values.flatten()

                    # Extract evidence variables for the current variable from the test data
                    evidence = list(original_bn.get_parents(variable))
                    test_data = test_df[evidence].values.astype('float32') if evidence else np.zeros((test_df.shape[0], 1)).astype('float32')

                    # NeuralBN CPD prediction using the test data
                    with torch.no_grad():
                        nn_cpd = neural_bn.models[variable](torch.tensor(test_data).to(device)).cpu().numpy().flatten()

                    # Traditional BN CPD prediction using the learned CPDs on the test data
                    table_cpd = traditional_bn.get_cpds(variable).values.flatten()

                    # Ensure all CPDs have the same shape by adjusting the shapes (padding to match largest)
                    max_len = max(len(original_cpd), len(nn_cpd), len(table_cpd))

                    if len(original_cpd) < max_len:
                        original_cpd = np.pad(original_cpd, (0, max_len - len(original_cpd)), 'constant', constant_values=1e-8)
                    
                    if len(nn_cpd) < max_len:
                        nn_cpd = np.pad(nn_cpd, (0, max_len - len(nn_cpd)), 'constant', constant_values=1e-8)
                    
                    if len(table_cpd) < max_len:
                        table_cpd = np.pad(table_cpd, (0, max_len - len(table_cpd)), 'constant', constant_values=1e-8)

                    # Apply smoothing to avoid zeros and ensure safe KL divergence calculation
                    original_cpd = smooth_distribution(original_cpd)
                    nn_cpd = smooth_distribution(nn_cpd)
                    table_cpd = smooth_distribution(table_cpd)

                    # Print the CPDs to debug the values
                    print(f"Original CPD for {variable}: {original_cpd}")
                    print(f"NeuralBN Predicted CPD for {variable}: {nn_cpd}")
                    print(f"TraditionalBN Predicted CPD for {variable}: {table_cpd}")

                    # Compute KL Divergence after smoothing
                    kl_div_nn = entropy(original_cpd, nn_cpd)
                    kl_div_table = entropy(original_cpd, table_cpd)
                    
                    return kl_div_nn, kl_div_table



                kl_divs = {}
                for variable in ['A', 'B', 'C']:
                    kl_div_nn, kl_div_table = compare_with_ground_truth(variable)
                    kl_divs[variable] = (kl_div_nn, kl_div_table)
                    print(f"\nKL Divergence for {variable} at dataset size {dataset_size}: NeuralBN = {kl_div_nn:.4f}, Traditional BN = {kl_div_table:.4f}")

                results.append((structure_name, dataset_size, iteration, kl_divs, neural_time, traditional_time))

# Organize the results into a DataFrame
data = []
for result in results:
    structure_name, dataset_size, iteration, kl_divs, neural_time, traditional_time = result
    for variable in kl_divs:
        kl_div_nn, kl_div_table = kl_divs[variable]
        data.append({
            'Structure': structure_name,
            'Dataset Size': dataset_size,
            'Iteration': iteration,
            'Variable': variable,
            'KL Divergence NeuralBN': kl_div_nn,
            'KL Divergence TraditionalBN': kl_div_table,
            'Training Time NeuralBN': neural_time,
            'Training Time TraditionalBN': traditional_time
        })

# After organizing results into a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file for future use
df.to_csv("experiment_results.csv", index=False)

print("DONE!!!")


