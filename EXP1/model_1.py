import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
import time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
import matplotlib.pyplot as plt
from pgmpy.estimators import MaximumLikelihoodEstimator


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


# ---------------- Neural Bayesian Network Definitions ---------------------
class CPDNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(CPDNetwork, self).__init__()
        # Randomly decide on the number of layers and units per layer
        layers_config = [np.random.choice([2,4,8,16,32]) for _ in range(np.random.choice([1, 2]))]
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

    def fit(self, data, epochs=60, batch_size=5, patience=10):
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


# ----------------- Traditional Bayesian Network with Data Splitting -----------------

def split_data(data_df, test_size=0.3, random_state=None):
    # Split the data into training and testing sets
    train_df, test_df = train_test_split(data_df, test_size=test_size, random_state=random_state)
    return train_df, test_df

def create_traditional_bn(train_df, structure):
    # Create and fit the Traditional Bayesian Network using the training data
    traditional_bn = BayesianNetwork(structure)
    traditional_bn.fit(train_df, estimator=MaximumLikelihoodEstimator)
    return traditional_bn

def evaluate_bn(traditional_bn, test_df, original_bn):
    kl_divergences = {}
    
    for node in traditional_bn.nodes():
        # Get the evidence variables for the current node
        evidence_vars = traditional_bn.get_parents(node)
        
        # Extract evidence values from the test set
        test_data = test_df[evidence_vars].values.astype('float32') if evidence_vars else np.zeros((test_df.shape[0], 1)).astype('float32')

        # Get the original CPD from the original BN
        original_cpd = original_bn.get_cpds(node).values.flatten()

        # Get the predicted CPD from the traditional BN
        predicted_cpd = traditional_bn.predict(test_df)[node].values.flatten()

        # Ensure CPDs have the same shape by padding
        max_len = max(len(original_cpd), len(predicted_cpd))

        if len(original_cpd) < max_len:
            original_cpd = np.pad(original_cpd, (0, max_len - len(original_cpd)), 'constant', constant_values=1e-8)
        if len(predicted_cpd) < max_len:
            predicted_cpd = np.pad(predicted_cpd, (0, max_len - len(predicted_cpd)), 'constant', constant_values=1e-8)

        # Apply smoothing to avoid zeros
        original_cpd = smooth_distribution(original_cpd)
        predicted_cpd = smooth_distribution(predicted_cpd)

        # Compute KL divergence
        kl_div = entropy(original_cpd, predicted_cpd)

        kl_divergences[node] = kl_div
    
    return kl_divergences


def smooth_distribution(distribution, epsilon=1e-5):
    # Apply smoothing by adding a small epsilon and normalize
    distribution = np.clip(distribution + epsilon, a_min=epsilon, a_max=None)
    return distribution / np.sum(distribution)

def compare_with_ground_truth(original_bn, traditional_bn, test_df, variable):
    # Get the ground truth CPD from the original BN
    original_cpd = original_bn.get_cpds(variable).values.flatten()

    # Extract the evidence variables for the current variable from the test data
    evidence = list(original_bn.get_parents(variable))
    test_data = test_df[evidence].values.astype('float32') if evidence else np.zeros((test_df.shape[0], 1)).astype('float32')

    # Traditional BN CPD prediction using the learned CPDs on the test data
    table_cpd = traditional_bn.get_cpds(variable).values.flatten()

    # Ensure all CPDs have the same shape by adjusting the shapes (padding to match largest)
    max_len = max(len(original_cpd), len(table_cpd))

    if len(original_cpd) < max_len:
        original_cpd = np.pad(original_cpd, (0, max_len - len(original_cpd)), 'constant', constant_values=1e-8)

    if len(table_cpd) < max_len:
        table_cpd = np.pad(table_cpd, (0, max_len - len(table_cpd)), 'constant', constant_values=1e-8)

    # Apply smoothing to avoid zeros and ensure safe KL divergence calculation
    original_cpd = smooth_distribution(original_cpd)
    table_cpd = smooth_distribution(table_cpd)

    # Compute KL Divergence after smoothing
    kl_div_table = entropy(original_cpd, table_cpd)

    return kl_div_table


# ---------------- CPD Random Generation ------------------

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
# ---------------- Main Execution ------------------

# Define the structures and dataset sizes for testing
dataset_sizes = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] 
iterations = 25

model_structures = {
    'chain_structure': [('A', 'B'), ('B', 'C')],
    'complex_structure': [('A', 'B'), ('B', 'C'), ('A', 'C')],
    'tree_structure': [('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'E'), ('C', 'F')]
}
# Initialize results list
results = []

# Loop over structures, dataset sizes, and iterations
for structure_name, structure in model_structures.items():
    for dataset_size in dataset_sizes:
        for iteration in range(iterations):
            print(f"Dataset Size: {dataset_size}, Iteration: {iteration + 1}")
            
            # Generate random CPDs and synthetic data
            cpds = generate_random_cpds(structure_name)
            data_df = generate_synthetic_data(structure, cpds, dataset_size)
            
            # Split data into train and test
            train_df, test_df = split_data(data_df, test_size=0.2, random_state=iteration)

            # ------------------ Train Neural Bayesian Network -------------------
            neural_bn = NeuralBayesianNetwork(structure)
            start_time = time.time()
            neural_bn.fit(train_df)
            neural_time = time.time() - start_time
            print(f"Training time for NeuralBN: {neural_time:.4f} seconds")

            # ------------------ Train Traditional Bayesian Network -------------------
            traditional_bn = create_traditional_bn(train_df, structure)
            
            # ------------------ Original Bayesian Network (for ground truth) -------------------
            original_bn = BayesianNetwork(structure)
            for cpd in cpds:
                original_bn.add_cpds(cpd)
            original_bn.check_model()

            # ------------------ Evaluate KL Divergences for Both Models -------------------
            kl_divs_neural_bn = {}
            kl_divs_traditional_bn = {}

            # Loop through each node to compare CPDs
            for node in traditional_bn.nodes():
                # Compare KL Divergence for NeuralBN
                kl_div_neural = compare_with_ground_truth(original_bn, neural_bn, test_df, node)
                kl_divs_neural_bn[node] = kl_div_neural

                # Compare KL Divergence for TraditionalBN
                kl_div_traditional = compare_with_ground_truth(original_bn, traditional_bn, test_df, node)
                kl_divs_traditional_bn[node] = kl_div_traditional

            # ------------------ Store Results for Each Node -------------------
            for node in traditional_bn.nodes():
                results.append({
                    'Structure': structure_name,
                    'Dataset Size': dataset_size,
                    'Iteration': iteration + 1,
                    'Node': node,
                    'KL Divergence NeuralBN': kl_divs_neural_bn[node],
                    'KL Divergence TraditionalBN': kl_divs_traditional_bn[node],
                    'Training Time NeuralBN': neural_time,
                })

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv("experiment_results.csv", index=False)
print("Results saved to experiment_results.csv.")
