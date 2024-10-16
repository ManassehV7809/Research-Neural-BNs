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
import matplotlib.pyplot as plt
from pgmpy.estimators import MaximumLikelihoodEstimator
from tqdm import tqdm

# Determine the device being used for PyTorch
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device for PyTorch")


# ---------------- Neural Bayesian Network Definitions ---------------------
class CPDNetwork(nn.Module):
    def __init__(self, input_size, output_size, layers_config):
        super(CPDNetwork, self).__init__()
        layers = []
        for units in layers_config:
            layers.append(nn.Linear(input_size, units))
            layers.append(nn.LayerNorm(units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_size = units
        layers.append(nn.Linear(input_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



class NeuralBayesianNetwork(BayesianNetwork):
    def __init__(self, ebunch=None):
        super(NeuralBayesianNetwork, self).__init__(ebunch)
        self.models = {}
        self.architecture_log = []

    def fit(self, data, epochs=60, batch_size=5, patience=10, fixed_architecture=None):
        for node in self.nodes():
            parents = list(self.get_parents(node))
            self._add_cpd_with_nn(
                node,
                parents,
                data,
                epochs,
                batch_size,
                patience,
                fixed_architecture,
            )
        self.check_model()

    def _add_cpd_with_nn(
        self,
        variable,
        evidence,
        data,
        epochs,
        batch_size,
        patience,
        fixed_architecture,
    ):
        X = (
            data[evidence].values.astype("float32")
            if evidence
            else np.zeros((data.shape[0], 1), dtype="float32")
        )
        y = data[variable].values
        input_size = X.shape[1] if evidence else 1

        # Calculate output size: always 2 rows (for binary variable)
        output_size = 2  # Binary outcome for the variable itself

        # Convert data to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(device)

        # Check if the target tensor contains only 0s and 1s
        assert torch.all(
            (y_tensor == 0) | (y_tensor == 1)
        ), f"Target tensor contains values other than 0 and 1 for variable {variable}"

        # Create and train the model with fixed architecture
        model, architecture_details = self._train_nn(
            X_tensor,
            y_tensor,
            input_size,
            output_size,
            epochs,
            batch_size,
            patience,
            fixed_architecture,variable
        )
        self.models[variable] = model
        self.architecture_log.append((variable, architecture_details))

        # Predict CPD values
        if evidence:
            evidence_values = [list(range(2)) for _ in evidence]  # Assuming binary variables
            evidence_combinations = np.array(
                np.meshgrid(*evidence_values)
            ).T.reshape(-1, input_size)
        else:
            evidence_combinations = np.array([[0]])  # No parents, so just a placeholder
        evidence_tensor = torch.tensor(evidence_combinations, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model(evidence_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_cpd = probabilities.cpu().numpy().T  # Shape: (output_size, num_columns)

        # Create and add TabularCPD
        if evidence:
            cpd = TabularCPD(
                variable=variable,
                variable_card=2,
                values=predicted_cpd,
                evidence=evidence,
                evidence_card=[2] * len(evidence),
                state_names={variable: [0, 1], **{e: [0, 1] for e in evidence}},
            )
        else:
            cpd = TabularCPD(
                variable=variable,
                variable_card=2,
                values=predicted_cpd,
                state_names={variable: [0, 1]},
            )

        # Add the CPD to the model
        self.add_cpds(cpd)

        
    def _train_nn(
    self,
    X_train,
    y_train,
    input_size,
    output_size,
    epochs,
    batch_size,
    patience,
    layers_config,
    variable=None):
        model = CPDNetwork(input_size, output_size, layers_config).to(device)
        criterion = nn.CrossEntropyLoss()
        # Try a lower learning rate
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        best_loss = float("inf")
        best_model_state = None
        early_stop_counter = 0

        # Initialize list to store training losses
        training_losses = []

        # Create Dataset and DataLoader
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)  # Multiply by batch size

            # Calculate average loss over the epoch
            avg_loss = epoch_loss / len(dataset)

            # Append average loss to the list
            training_losses.append(avg_loss)

            # Early stopping logic
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = model.state_dict()
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        architecture_details = {
            "layers_config": layers_config,
            "loss": best_loss,
        }

        # # Plot the training losses
        # import matplotlib.pyplot as plt

        # plt.figure()
        # plt.plot(training_losses, label='Training Loss')
        # plt.title(f'Training Loss for Variable {variable}')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.show()

        return model, architecture_details


    def sample(self, n):
        sampler = BayesianModelSampling(self)
        samples = sampler.forward_sample(size=n)
        return samples


# ----------------- Traditional Bayesian Network with Data Splitting -----------------
def split_data(data_df, test_size=0.3, random_state=None):
    train_df, test_df = train_test_split(
        data_df, test_size=test_size, random_state=random_state
    )
    train_df = train_df.astype(int)
    test_df = test_df.astype(int)
    return train_df, test_df


def create_traditional_bn(train_df, structure):
    traditional_bn = BayesianNetwork(structure)
    traditional_bn.fit(
        train_df,
        estimator=MaximumLikelihoodEstimator,
        state_names={var: [0, 1] for var in train_df.columns},
    )
    return traditional_bn


# ---------------------------KL Divergences comparison ----------------------------#
def smooth_distribution(distribution, epsilon=1e-5):
    # Apply smoothing by adding a small epsilon and normalize
    distribution = np.clip(distribution + epsilon, a_min=epsilon, a_max=None)
    return distribution / np.sum(distribution)


def compare_with_ground_truth(original_bn, neural_bn, traditional_bn, test_df, variable):
    # Get the ground truth CPD from the original BN
    original_cpd = original_bn.get_cpds(variable)

    # Get the evidence variables
    evidence = list(original_bn.get_parents(variable))

    # Generate all possible evidence configurations from test_df
    if evidence:
        evidence_values = [test_df[e].unique() for e in evidence]
        evidence_combinations = np.array(np.meshgrid(*evidence_values)).T.reshape(-1, len(evidence))
    else:
        evidence_combinations = np.array([[0]])  # No parents, so just a placeholder

    kl_div_nn_total = 0.0
    kl_div_table_total = 0.0
    num_configurations = evidence_combinations.shape[0]

    for i in range(num_configurations):
        evidence_config = evidence_combinations[i]
        evidence_tuples = list(zip(evidence, evidence_config))

        # Get ground truth CPD for this evidence configuration
        original_probs = original_cpd.reduce(evidence_tuples, inplace=False).values.flatten()

        # Neural BN prediction
        if evidence:
            X_nn = torch.tensor(evidence_config.astype("float32")).unsqueeze(0).to(device)
        else:
            X_nn = torch.tensor([[0]], dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs_nn = neural_bn.models[variable](X_nn)
            probs_nn = torch.softmax(outputs_nn, dim=1).cpu().numpy().flatten()

        # Traditional BN prediction
        cpd_traditional = traditional_bn.get_cpds(variable)
        probs_traditional = cpd_traditional.reduce(
            evidence_tuples, inplace=False
        ).values.flatten()

        # Apply smoothing
        original_probs = smooth_distribution(original_probs)
        probs_nn = smooth_distribution(probs_nn)
        probs_traditional = smooth_distribution(probs_traditional)

        # Compute KL divergence
        kl_div_nn = entropy(original_probs, probs_nn)
        kl_div_table = entropy(original_probs, probs_traditional)

        kl_div_nn_total += kl_div_nn
        kl_div_table_total += kl_div_table

    # Average KL divergence over all evidence configurations
    kl_div_nn_avg = kl_div_nn_total / num_configurations
    kl_div_table_avg = kl_div_table_total / num_configurations

    return kl_div_nn_avg, kl_div_table_avg


# ---------------------- Random CPD Generation ---------------------------#
def generate_random_cpds(structure_name):
    if structure_name == "chain_structure":
        # CPD for A (no parents, binary)
        prob_A = np.random.rand()
        cpd_A = TabularCPD(
            variable="A",
            variable_card=2,
            values=[[prob_A], [1 - prob_A]],
            state_names={"A": [0, 1]},
        )

        # CPD for B with parent A (2x2 slots)
        prob_B = np.random.rand(2, 2)
        prob_B = prob_B / prob_B.sum(axis=0, keepdims=True)  # Normalize columns
        cpd_B = TabularCPD(
            variable="B",
            variable_card=2,
            values=prob_B,
            evidence=["A"],
            evidence_card=[2],
            state_names={"B": [0, 1], "A": [0, 1]},
        )

        # CPD for C with parent B (2x2 slots)
        prob_C = np.random.rand(2, 2)
        prob_C = prob_C / prob_C.sum(axis=0, keepdims=True)  # Normalize columns
        cpd_C = TabularCPD(
            variable="C",
            variable_card=2,
            values=prob_C,
            evidence=["B"],
            evidence_card=[2],
            state_names={"C": [0, 1], "B": [0, 1]},
        )

        return [cpd_A, cpd_B, cpd_C]

    elif structure_name == "triangular_structure":
        # CPD for A
        prob_A = np.random.rand()
        cpd_A = TabularCPD(
            variable="A",
            variable_card=2,
            values=[[prob_A], [1 - prob_A]],
            state_names={"A": [0, 1]},
        )

        # CPD for B with parent A
        prob_B = np.random.rand(2, 2)
        prob_B = prob_B / prob_B.sum(axis=0, keepdims=True)
        cpd_B = TabularCPD(
            variable="B",
            variable_card=2,
            values=prob_B,
            evidence=["A"],
            evidence_card=[2],
            state_names={"B": [0, 1], "A": [0, 1]},
        )

        # CPD for C with parents A and B (2x4 slots)
        prob_C = np.random.rand(2, 4)
        prob_C = prob_C / prob_C.sum(axis=0, keepdims=True)
        cpd_C = TabularCPD(
            variable="C",
            variable_card=2,
            values=prob_C,
            evidence=["A", "B"],
            evidence_card=[2, 2],
            state_names={"C": [0, 1], "A": [0, 1], "B": [0, 1]},
        )

        return [cpd_A, cpd_B, cpd_C]

    elif structure_name == "tree_structure":
        # Nodes: A, B, C, D, E, F, G, H, I, J, K

        # CPD for A (root node, no parents)
        prob_A = np.random.rand()
        cpd_A = TabularCPD(
            variable="A",
            variable_card=2,
            values=[[prob_A], [1 - prob_A]],
            state_names={"A": [0, 1]},
        )

        # CPD for B with parent A
        prob_B = np.random.rand(2, 2)
        prob_B = prob_B / prob_B.sum(axis=0, keepdims=True)
        cpd_B = TabularCPD(
            variable="B",
            variable_card=2,
            values=prob_B,
            evidence=["A"],
            evidence_card=[2],
            state_names={"B": [0, 1], "A": [0, 1]},
        )

        # CPD for C with parent A
        prob_C = np.random.rand(2, 2)
        prob_C = prob_C / prob_C.sum(axis=0, keepdims=True)
        cpd_C = TabularCPD(
            variable="C",
            variable_card=2,
            values=prob_C,
            evidence=["A"],
            evidence_card=[2],
            state_names={"C": [0, 1], "A": [0, 1]},
        )

        # CPD for D with parent B
        prob_D = np.random.rand(2, 2)
        prob_D = prob_D / prob_D.sum(axis=0, keepdims=True)
        cpd_D = TabularCPD(
            variable="D",
            variable_card=2,
            values=prob_D,
            evidence=["B"],
            evidence_card=[2],
            state_names={"D": [0, 1], "B": [0, 1]},
        )

        # CPD for E with parent B
        prob_E = np.random.rand(2, 2)
        prob_E = prob_E / prob_E.sum(axis=0, keepdims=True)
        cpd_E = TabularCPD(
            variable="E",
            variable_card=2,
            values=prob_E,
            evidence=["B"],
            evidence_card=[2],
            state_names={"E": [0, 1], "B": [0, 1]},
        )

        # CPD for F with parent C
        prob_F = np.random.rand(2, 2)
        prob_F = prob_F / prob_F.sum(axis=0, keepdims=True)
        cpd_F = TabularCPD(
            variable="F",
            variable_card=2,
            values=prob_F,
            evidence=["C"],
            evidence_card=[2],
            state_names={"F": [0, 1], "C": [0, 1]},
        )

        # CPD for G with parent C
        prob_G = np.random.rand(2, 2)
        prob_G = prob_G / prob_G.sum(axis=0, keepdims=True)
        cpd_G = TabularCPD(
            variable="G",
            variable_card=2,
            values=prob_G,
            evidence=["C"],
            evidence_card=[2],
            state_names={"G": [0, 1], "C": [0, 1]},
        )

        # CPD for H with parent D
        prob_H = np.random.rand(2, 2)
        prob_H = prob_H / prob_H.sum(axis=0, keepdims=True)
        cpd_H = TabularCPD(
            variable="H",
            variable_card=2,
            values=prob_H,
            evidence=["D"],
            evidence_card=[2],
            state_names={"H": [0, 1], "D": [0, 1]},
        )

        # CPD for I with parent E
        prob_I = np.random.rand(2, 2)
        prob_I = prob_I / prob_I.sum(axis=0, keepdims=True)
        cpd_I = TabularCPD(
            variable="I",
            variable_card=2,
            values=prob_I,
            evidence=["E"],
            evidence_card=[2],
            state_names={"I": [0, 1], "E": [0, 1]},
        )

        # CPD for J with parent F
        prob_J = np.random.rand(2, 2)
        prob_J = prob_J / prob_J.sum(axis=0, keepdims=True)
        cpd_J = TabularCPD(
            variable="J",
            variable_card=2,
            values=prob_J,
            evidence=["F"],
            evidence_card=[2],
            state_names={"J": [0, 1], "F": [0, 1]},
        )

        # CPD for K with parent G
        prob_K = np.random.rand(2, 2)
        prob_K = prob_K / prob_K.sum(axis=0, keepdims=True)
        cpd_K = TabularCPD(
            variable="K",
            variable_card=2,
            values=prob_K,
            evidence=["G"],
            evidence_card=[2],
            state_names={"K": [0, 1], "G": [0, 1]},
        )

        return [cpd_A, cpd_B, cpd_C, cpd_D, cpd_E, cpd_F, cpd_G, cpd_H, cpd_I, cpd_J, cpd_K]

    elif structure_name == "sparse_structure":
        # Nodes: A, B, C, D, E, F, G, H

        # CPD for A
        prob_A = np.random.rand()
        cpd_A = TabularCPD(
            variable="A",
            variable_card=2,
            values=[[prob_A], [1 - prob_A]],
            state_names={"A": [0, 1]},
        )

        # CPD for B
        prob_B = np.random.rand()
        cpd_B = TabularCPD(
            variable="B",
            variable_card=2,
            values=[[prob_B], [1 - prob_B]],
            state_names={"B": [0, 1]},
        )

        # CPD for C
        prob_C = np.random.rand()
        cpd_C = TabularCPD(
            variable="C",
            variable_card=2,
            values=[[prob_C], [1 - prob_C]],
            state_names={"C": [0, 1]},
        )

        # CPD for D
        prob_D = np.random.rand()
        cpd_D = TabularCPD(
            variable="D",
            variable_card=2,
            values=[[prob_D], [1 - prob_D]],
            state_names={"D": [0, 1]},
        )

        # CPD for E with parents A, B, C
        prob_E = np.random.rand(2, 8)  # 2 rows, 2^3=8 columns
        prob_E = prob_E / prob_E.sum(axis=0, keepdims=True)
        cpd_E = TabularCPD(
            variable="E",
            variable_card=2,
            values=prob_E,
            evidence=["A", "B", "C"],
            evidence_card=[2, 2, 2],
            state_names={"E": [0, 1], "A": [0, 1], "B": [0, 1], "C": [0, 1]},
        )

        # CPD for F with parent D
        prob_F = np.random.rand(2, 4)
        prob_F = prob_F / prob_F.sum(axis=0, keepdims=True)
        cpd_F = TabularCPD(
            variable="F",
            variable_card=2,
            values=prob_F,
            evidence=["D","E"],
            evidence_card=[2,2],
            state_names={"F": [0, 1], "D": [0, 1],"E": [0, 1]},
        )

        # CPD for G with parent E
        prob_G = np.random.rand(2, 2)
        prob_G = prob_G / prob_G.sum(axis=0, keepdims=True)
        cpd_G = TabularCPD(
            variable="G",
            variable_card=2,
            values=prob_G,
            evidence=["E"],
            evidence_card=[2],
            state_names={"G": [0, 1], "E": [0, 1]},
        )

        # CPD for H with parents F and G (2x4 slots)
        prob_H = np.random.rand(2, 4)
        prob_H = prob_H / prob_H.sum(axis=0, keepdims=True)
        cpd_H = TabularCPD(
            variable="H",
            variable_card=2,
            values=prob_H,
            evidence=["F", "G"],
            evidence_card=[2, 2],
            state_names={"H": [0, 1], "F": [0, 1], "G": [0, 1]},
        )

        return [cpd_A, cpd_B, cpd_C, cpd_D, cpd_E, cpd_F, cpd_G, cpd_H]

    elif structure_name == "sparse_structure_2":
        # Nodes: A, B, C, D, E, F, G, H, I, J

        # CPD for A
        prob_A = np.random.rand()
        cpd_A = TabularCPD(
            variable="A",
            variable_card=2,
            values=[[prob_A], [1 - prob_A]],
            state_names={"A": [0, 1]},
        )

        # CPD for B
        prob_B = np.random.rand()
        cpd_B = TabularCPD(
            variable="B",
            variable_card=2,
            values=[[prob_B], [1 - prob_B]],
            state_names={"B": [0, 1]},
        )

        # CPD for C
        prob_C = np.random.rand()
        cpd_C = TabularCPD(
            variable="C",
            variable_card=2,
            values=[[prob_C], [1 - prob_C]],
            state_names={"C": [0, 1]},
        )

        # CPD for D
        prob_D = np.random.rand()
        cpd_D = TabularCPD(
            variable="D",
            variable_card=2,
            values=[[prob_D], [1 - prob_D]],
            state_names={"D": [0, 1]},
        )

        # CPD for E 
        prob_E = np.random.rand()
        cpd_E = TabularCPD(
            variable="E",
            variable_card=2,
            values=[[prob_E], [1 - prob_E]],
            state_names={"E": [0, 1]},
        )

        # CPD for F 
        prob_F = np.random.rand()
        cpd_F = TabularCPD(
            variable="F",
            variable_card=2,
            values=[[prob_F], [1 - prob_F]],
            state_names={"F": [0, 1]},
        )


        # CPD for G with parentS A B C
        prob_G = np.random.rand(2, 8)
        prob_G = prob_G / prob_G.sum(axis=0, keepdims=True)
        cpd_G = TabularCPD(
            variable="G",
            variable_card=2,
            values=prob_G,
            evidence=["A", "B","C"],
            evidence_card=[2,2,2],
            state_names={"G": [0, 1], "A": [0, 1],"B": [0, 1],"C": [0, 1]},
        )

        # CPD for H with parent E 
        prob_H = np.random.rand(2, 4)
        prob_H = prob_H / prob_H.sum(axis=0, keepdims=True)
        cpd_H = TabularCPD(
            variable="H",
            variable_card=2,
            values=prob_H,
            evidence=["E","D"],
            evidence_card=[2,2],
            state_names={"H": [0, 1], "E": [0, 1], "D": [0, 1]},
        )

        # CPD for I with parentS H and G
        prob_I = np.random.rand(2, 4)
        prob_I = prob_I / prob_I.sum(axis=0, keepdims=True)
        cpd_I = TabularCPD(
            variable="I",
            variable_card=2,
            values=prob_I,
            evidence=["H","G"],
            evidence_card=[2,2],
            state_names={"I": [0, 1], "H": [0, 1], "G": [0, 1]},
        )

        # CPD for J with parents F, H, I (2x8 slots)
        prob_J = np.random.rand(2, 8)
        prob_J = prob_J / prob_J.sum(axis=0, keepdims=True)
        cpd_J = TabularCPD(
            variable="J",
            variable_card=2,
            values=prob_J,
            evidence=["F", "H", "I"],
            evidence_card=[2, 2, 2],
            state_names={"J": [0, 1], "F": [0, 1], "H": [0, 1], "I": [0, 1]},
        )

        return [cpd_A, cpd_B, cpd_C, cpd_D, cpd_E, cpd_F, cpd_G, cpd_H, cpd_I, cpd_J]
    
    elif structure_name == "dense_structure":
        # CPD for A (no parents, binary)
        prob_A = np.random.rand()
        cpd_A = TabularCPD(variable='A', variable_card=2, values=[[prob_A], [1 - prob_A]])
        
        # CPD for B (no parents, binary)
        prob_B = np.random.rand()
        cpd_B = TabularCPD(variable='B', variable_card=2, values=[[prob_B], [1 - prob_B]])
        
        # CPD for C (no parents, binary)
        prob_C = np.random.rand()
        cpd_C = TabularCPD(variable='C', variable_card=2, values=[[prob_C], [1 - prob_C]])
        
        # CPD for D with parents A, B (2x4 slots)
        prob_D = np.random.rand(2, 4)
        prob_D /= prob_D.sum(axis=0, keepdims=True)
        cpd_D = TabularCPD(
            variable='D',
            variable_card=2,
            values=prob_D,
            evidence=['A', 'B'],
            evidence_card=[2, 2],
            state_names={"D": [0, 1], "A": [0,1], "B": [0,1]}
        )
        
        # CPD for E with parents A, C (2x4 slots)
        prob_E = np.random.rand(2, 4)
        prob_E /= prob_E.sum(axis=0, keepdims=True)
        cpd_E = TabularCPD(
            variable='E',
            variable_card=2,
            values=prob_E,
            evidence=['A', 'C'],
            evidence_card=[2, 2],
            state_names={"E": [0, 1], "A": [0,1], "C": [0,1]}
        )
        
        # CPD for F with parent A (2x2 slots)
        prob_F = np.random.rand(2, 2)
        prob_F /= prob_F.sum(axis=0, keepdims=True)
        cpd_F = TabularCPD(
            variable='F',
            variable_card=2,
            values=prob_F,
            evidence=['A'],
            evidence_card=[2],
            state_names={"F": [0, 1], "A": [0,1]}
        )
        
        # CPD for G with parents B, C (2x4 slots)
        prob_G = np.random.rand(2, 4)
        prob_G /= prob_G.sum(axis=0, keepdims=True)
        cpd_G = TabularCPD(
            variable='G',
            variable_card=2,
            values=prob_G,
            evidence=['B', 'C'],
            evidence_card=[2, 2],
            state_names={"G": [0, 1], "B": [0,1], "C": [0,1]}
        )
        
        # CPD for H with parent B (2x2 slots)
        prob_H = np.random.rand(2, 2)
        prob_H /= prob_H.sum(axis=0, keepdims=True)
        cpd_H = TabularCPD(
            variable='H',
            variable_card=2,
            values=prob_H,
            evidence=['B'],
            evidence_card=[2],
            state_names={"H": [0, 1], "B": [0,1]}
        )
        
        # CPD for I with parent C (2x2 slots)
        prob_I = np.random.rand(2, 2)
        prob_I /= prob_I.sum(axis=0, keepdims=True)
        cpd_I = TabularCPD(
            variable='I',
            variable_card=2,
            values=prob_I,
            evidence=['C'],
            evidence_card=[2],
            state_names={"I": [0, 1], "C": [0,1]}
        )
        
        # CPD for J with parents D, E (2x4 slots)
        prob_J = np.random.rand(2, 4)
        prob_J /= prob_J.sum(axis=0, keepdims=True)
        cpd_J = TabularCPD(
            variable='J',
            variable_card=2,
            values=prob_J,
            evidence=['D', 'E'],
            evidence_card=[2, 2],
            state_names={"J": [0, 1], "D": [0,1], "E": [0,1]}
        )
        
        # CPD for K with parents D, F (2x4 slots)
        prob_K = np.random.rand(2, 4)
        prob_K /= prob_K.sum(axis=0, keepdims=True)
        cpd_K = TabularCPD(
            variable='K',
            variable_card=2,
            values=prob_K,
            evidence=['D', 'F'],
            evidence_card=[2, 2],
            state_names={"K": [0, 1], "D": [0,1], "F": [0,1]}
        )
        
        # CPD for L with parents E, G (2x4 slots)
        prob_L = np.random.rand(2, 4)
        prob_L /= prob_L.sum(axis=0, keepdims=True)
        cpd_L = TabularCPD(
            variable='L',
            variable_card=2,
            values=prob_L,
            evidence=['E', 'G'],
            evidence_card=[2, 2],
            state_names={"L": [0, 1], "E": [0,1], "G": [0,1]}
        )
        
        # CPD for M with parents F, H (2x4 slots)
        prob_M = np.random.rand(2, 4)
        prob_M /= prob_M.sum(axis=0, keepdims=True)
        cpd_M = TabularCPD(
            variable='M',
            variable_card=2,
            values=prob_M,
            evidence=['F', 'H'],
            evidence_card=[2, 2],
            state_names={"M": [0, 1], "F": [0,1], "H": [0,1]}
        )
        
        # CPD for N with parents G, I (2x4 slots)
        prob_N = np.random.rand(2, 4)
        prob_N /= prob_N.sum(axis=0, keepdims=True)
        cpd_N = TabularCPD(
            variable='N',
            variable_card=2,
            values=prob_N,
            evidence=['G', 'I'],
            evidence_card=[2, 2],
            state_names={"N": [0, 1], "G": [0,1], "I": [0,1]}
        )
        
        # CPD for O with parent H
        prob_O = np.random.rand(2, 2)
        prob_O /= prob_O.sum(axis=0, keepdims=True)
        cpd_O = TabularCPD(
            variable='O',
            variable_card=2,
            values=prob_O,
            evidence=['H'],
            evidence_card=[2],
            state_names={"O": [0, 1], "H": [0,1]}
        )
        
        # CPD for P with parent I
        prob_P = np.random.rand(2, 2)
        prob_P /= prob_P.sum(axis=0, keepdims=True)
        cpd_P = TabularCPD(
            variable='P',
            variable_card=2,
            values=prob_P,
            evidence=['I'],
            evidence_card=[2],
            state_names={"P": [0, 1], "I": [0,1]}
        )
        
        # CPD for Q with parents J, K (2x4 slots)
        prob_Q = np.random.rand(2, 4)
        prob_Q /= prob_Q.sum(axis=0, keepdims=True)
        cpd_Q = TabularCPD(
            variable='Q',
            variable_card=2,
            values=prob_Q,
            evidence=['J', 'K'],
            evidence_card=[2, 2],
            state_names={"Q": [0, 1], "J": [0,1], "K": [0,1]}
        )
        
        # CPD for R with parents L, M (2x4 slots)
        prob_R = np.random.rand(2, 4)
        prob_R /= prob_R.sum(axis=0, keepdims=True)
        cpd_R = TabularCPD(
            variable='R',
            variable_card=2,
            values=prob_R,
            evidence=['L', 'M'],
            evidence_card=[2, 2],
            state_names={"R": [0, 1], "L": [0,1], "M": [0,1]}
        )
        
        # CPD for S with parents N, O (2x4 slots)
        prob_S = np.random.rand(2, 4)
        prob_S /= prob_S.sum(axis=0, keepdims=True)
        cpd_S = TabularCPD(
            variable='S',
            variable_card=2,
            values=prob_S,
            evidence=['N', 'O'],
            evidence_card=[2, 2],
            state_names={"S": [0, 1], "N": [0,1], "O": [0,1]}
        )
        
        # CPD for T with parents P, Q, R (2x8 slots)
        prob_T = np.random.rand(2, 8)
        prob_T /= prob_T.sum(axis=0, keepdims=True)
        cpd_T = TabularCPD(
            variable='T',
            variable_card=2,
            values=prob_T,
            evidence=['P', 'Q', 'R'],
            evidence_card=[2, 2, 2],
            state_names={"T": [0, 1], "P": [0,1], "Q": [0,1], "R": [0,1]}
        )
        
        # CPD for U with parents Q, R, S (2x8 slots)
        prob_U = np.random.rand(2, 8)
        prob_U /= prob_U.sum(axis=0, keepdims=True)
        cpd_U = TabularCPD(
            variable='U',
            variable_card=2,
            values=prob_U,
            evidence=['Q', 'R', 'S'],
            evidence_card=[2, 2, 2],
            state_names={'U': [0, 1], 'Q': [0, 1], 'R': [0, 1], 'S': [0, 1]}
        )
        
        return [
            cpd_A, cpd_B, cpd_C, cpd_D, cpd_E, cpd_F, cpd_G, cpd_H,
            cpd_I, cpd_J, cpd_K, cpd_L, cpd_M, cpd_N, cpd_O, cpd_P,
            cpd_Q, cpd_R, cpd_S, cpd_T, cpd_U
        ]


    else:
        raise ValueError(f"Structure name '{structure_name}' not recognized.")




def generate_synthetic_data(structure, cpds, size):
    bn = BayesianNetwork(structure)
    for cpd in cpds:
        bn.add_cpds(cpd)
    bn.check_model()
    sampler = BayesianModelSampling(bn)
    data = sampler.forward_sample(size=size)
    return pd.DataFrame(data).astype(int)


def get_dynamic_batch_size(dataset_size):
    if dataset_size <= 10:
        return 1
    elif dataset_size <= 20:
        return 2
    elif dataset_size <= 50:
        return 4
    elif dataset_size <= 100:
        return 5
    elif dataset_size <= 500:
        return 32
    else:
        return 128
    

def experiment(fixed_layers_config, fixed_cpds, structure, structure_name, experiment_num):
    dataset_sizes = [i for i in range(5, 2000, 200)]
    iterations = 25
    results = []

    # Create the original BN to use as ground truth for KL divergence comparison
    original_bn = BayesianNetwork(structure)
    for cpd in fixed_cpds:
        original_bn.add_cpds(cpd)
    original_bn.check_model()

    for dataset_size in dataset_sizes:
        batch_size = get_dynamic_batch_size(dataset_size)  # Use dynamic batch size based on dataset size
        for iteration in range(iterations):
            print(f"Dataset Size: {dataset_size}, Iteration: {iteration + 1}, Batch Size: {batch_size}")
            data_df = generate_synthetic_data(structure, fixed_cpds, dataset_size)
            train_df, test_df = split_data(data_df, test_size=0.3, random_state=iteration)

            # Train NeuralBN with fixed architecture
            neural_bn = NeuralBayesianNetwork(structure)
            start_time = time.time()
            neural_bn.fit(train_df, batch_size=batch_size, fixed_architecture=fixed_layers_config)
            neural_time = time.time() - start_time

            # Train TraditionalBN
            traditional_bn = create_traditional_bn(train_df, structure)

            # Compare KL Divergence using test_df
            for node in traditional_bn.nodes():
                kl_div_neural, kl_div_traditional = compare_with_ground_truth(
                    original_bn, neural_bn, traditional_bn, test_df, node
                )
                results.append(
                    {
                        "Structure": structure_name,
                        "Experiment": experiment_num/2,  # Experiment number starts from 1
                        "Dataset Size": dataset_size,
                        "Node": node,
                        "Iteration": iteration + 1,
                        "KL Divergence NeuralBN": kl_div_neural,
                        "KL Divergence TraditionalBN": kl_div_traditional,
                        "Training Time NeuralBN": neural_time,
                    }
                )

    # Save results for this experiment
    df = pd.DataFrame(results)
    df.to_csv(f"{structure_name}_experiment_{experiment_num/2}_results.csv", index=False)
    print(f"Results saved for {structure_name} Experiment {experiment_num + 1}")

# ---------------- Run Experiments for All Structures ------------------
structures = {
    "chain_structure": [("A", "B"), ("B", "C")],
    "triangular_structure": [("A", "B"), ("B", "C"), ("A", "C")],
"tree_structure": [
    ("A", "B"),
    ("A", "C"),
    ("B", "D"),
    ("B", "E"),
    ("C", "F"),
    ("C", "G"),
    ("D", "H"),
    ("E", "I"),
    ("F", "J"),
    ("G", "K"),
    
]

,
"sparse_structure": [
    ("A", "E"),
    ("B", "E"),
    ("C", "E"),  
    ("D", "F"),
    ("E", "F"),
    ("E", "G"),
    ("F", "H"),
    ("G", "H"),
]
,
"sparse_structure_2": [
    ("A", "G"),
    ("B", "G"),
    ("C", "G"),
    ("D", "H"),
    ("G", "I"),
    ("H", "I"),
    ("E", "H"),
    ("I", "J"),
    ("H", "J"),
    ("F", "J"),
],


 "dense_structure" : [
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
        ('Q', 'U'),
        ('R', 'U'),
        ('S', 'U'),
    ]

}

layers_list = [2,4,6]
for structure_name, structure in structures.items():
    for experiment_num in layers_list:  # Run 3 experiments
        print(f"Running {structure_name} - Experiment {experiment_num/2}")
        # Generate fixed architecture (2 layer, 4 layers, and 6 layers for each experiment)
        fixed_layers_config = [
    np.random.choice([32, 64, 128]) for _ in range(experiment_num)]

        # Generate fixed CPDs for the experiment
        fixed_cpds = generate_random_cpds(structure_name)
        experiment(fixed_layers_config, fixed_cpds, structure, structure_name, experiment_num)

