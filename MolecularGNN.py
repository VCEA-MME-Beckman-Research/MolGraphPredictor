# -*- coding: utf-8 -*-
"""
This script implements a Graph Neural Network (GNN) for predicting molecular properties.
It uses RDKit for molecule processing, TensorFlow for building the GNN model, and Matplotlib for plotting results.
"""

import pickle
import rdkit
import rdkit.Chem
import rdkit.Chem.rdDepictor
import rdkit.Chem.Draw
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import os

# --- Configuration ---
# The number of features for each atom.
# We use a one-hot encoding of the atomic number, and since we have atoms up to Beryllium (atomic number 4),
# and we also have C, N, O, F, we need a vector of size 9 to represent them.
# H=1, He=2, Li=3, Be=4, B=5, C=6, N=7, O=8, F=9
N_FEATURES = 9
DATA_FILE = 'GCNN-data-2.pickle'
MODEL_SAVE_PATH = 'model/molecular_gnn.h5'

# --- Data Loading and Preprocessing ---

def gen_smiles2graph(sml):
    """
    Converts a SMILES string to a graph representation (nodes and adjacency matrix).

    Args:
        sml (str): A SMILES (Simplified Molecular-Input Line-Entry System) string,
                   which is a textual representation of a molecule's structure.

    Returns:
        tuple: A tuple containing:
            - nodes (np.ndarray): A matrix where each row represents an atom and its features (one-hot encoded atomic number).
            - adj (np.ndarray): An adjacency matrix representing the bonds between atoms.
    """
    # Convert the SMILES string to an RDKit molecule object.
    m = rdkit.Chem.MolFromSmiles(sml)
    # Add hydrogens to the molecule, as they are often implicit in SMILES strings.
    m = rdkit.Chem.AddHs(m)

    # Define a mapping from RDKit bond types to numerical values.
    order_string = {
        rdkit.Chem.rdchem.BondType.SINGLE: 1,
        rdkit.Chem.rdchem.BondType.DOUBLE: 2,
        rdkit.Chem.rdchem.BondType.TRIPLE: 3,
        rdkit.Chem.rdchem.BondType.AROMATIC: 4,
    }

    # Get the number of atoms in the molecule.
    N = len(list(m.GetAtoms()))
    # Initialize a matrix to store the node features.
    nodes = np.zeros((N, N_FEATURES))
    # Create the node features. Each atom is represented by a one-hot encoded vector
    # of its atomic number.
    for i in m.GetAtoms():
        nodes[i.GetIdx(), i.GetAtomicNum() -1] = 1 # We use atomic number - 1 as index

    # Initialize an adjacency matrix to represent the bonds.
    adj = np.zeros((N, N))
    # Populate the adjacency matrix based on the bonds in the molecule.
    for j in m.GetBonds():
        u = min(j.GetBeginAtomIdx(), j.GetEndAtomIdx())
        v = max(j.GetBeginAtomIdx(), j.GetEndAtomIdx())
        order = j.GetBondType()
        if order in order_string:
            # We are not using the bond order in this version of the model,
            # but this is where you would use it.
            pass
        else:
            raise Warning("Ignoring bond order" + str(order))
        # Set the adjacency matrix entries to 1 to indicate a bond.
        # The matrix is symmetric because bonds are bidirectional.
        adj[u, v] = 1
        adj[v, u] = 1
    # Add self-loops to the adjacency matrix. This is a common practice in GNNs
    # to allow a node to consider its own features during message passing.
    adj += np.eye(N)
    return nodes, adj

# --- GNN Model Definition ---

class GCNLayer(tf.keras.layers.Layer):
    """
    A Graph Convolutional Network (GCN) layer.

    This layer implements the core operation of a GCN. It updates the features of each node
    by aggregating information from its neighbors. The mathematical operation is:
    H' = D^(-1/2) * A * D^(-1/2) * H * W
    where:
    - H is the matrix of node features.
    - A is the adjacency matrix.
    - D is the degree matrix (a diagonal matrix of node degrees).
    - W is a trainable weight matrix.
    - H' is the updated matrix of node features.

    This implementation uses a simplified version:
    H' = D^(-1) * A * H * W
    """

    def __init__(self, activation=None, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        """Create the trainable weights for the layer."""
        node_shape, adj_shape = input_shape
        self.w = self.add_weight(shape=(node_shape[2], node_shape[2]), name="w")

    def call(self, inputs):
        """The forward pass of the layer."""
        nodes, adj = inputs
        # Compute the degree of each node by summing the rows of the adjacency matrix.
        degree = tf.reduce_sum(adj, axis=-1)
        # The GCN equation. `tf.einsum` is used for efficient tensor multiplication.
        # The equation effectively does the following:
        # 1. `tf.einsum("bij,bjk->bik", adj, nodes)`: For each graph in the batch, it multiplies
        #    the adjacency matrix `adj` with the node features `nodes`. This aggregates the
        #    features of the neighboring nodes.
        # 2. `tf.einsum("bik,kl->bil", ..., self.w)`: It then multiplies the result by the
        #    trainable weight matrix `w` to transform the aggregated features.
        # 3. `tf.einsum("bi,...->bi...", 1 / degree, ...)`: Finally, it normalizes the
        #    features by dividing by the degree of each node.
        new_nodes = tf.einsum("bi,bij,bjk,kl->bil", 1 / degree, adj, nodes, self.w)
        # Apply the activation function.
        out = self.activation(new_nodes)
        return out, adj

class GRLayer(tf.keras.layers.Layer):
    """
    A Graph Readout Layer.

    This layer aggregates the node features of a graph into a single graph-level
    representation. This is necessary to make a prediction for the entire molecule.
    This implementation simply sums the features of all nodes.
    """

    def __init__(self, name="GRLayer", **kwargs):
        super(GRLayer, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        nodes, adj = inputs
        # Sum the node features along the node axis to get a single vector for the graph.
        reduction = tf.reduce_sum(nodes, axis=1)
        return reduction

def build_model():
    """Builds the GNN model."""
    ninput = tf.keras.Input((None, N_FEATURES))
    ainput = tf.keras.Input((None, None))

    # GCN block: A stack of GCN layers to learn node representations.
    x = GCNLayer("relu")([ninput, ainput])
    x = GCNLayer("relu")(x)
    x = GCNLayer("relu")(x)
    x = GCNLayer("relu")(x)

    # Readout layer: Aggregates node features to a graph-level representation.
    x = GRLayer()(x)

    # Standard dense layers (the "readout" function) to make the final prediction.
    x = tf.keras.layers.Dense(16, "tanh")(x)
    x = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=(ninput, ainput), outputs=x)
    return model

# --- Data Generator ---

def data_generator(raw_data, first_moment_list):
    """
    A generator function that yields graph data and corresponding labels.
    """
    k = 0
    for i in raw_data['smiles'].values():
        graph = gen_smiles2graph(i)
        mom = first_moment_list[k]
        k += 1
        yield graph, mom

# --- Main Execution ---

def main():
    """Main function to run the GNN model."""
    # Load the raw data from the pickle file.
    # This file contains SMILES strings and their corresponding molecular properties.
    with open(DATA_FILE, 'rb') as f:
        raw_data = pickle.load(f)

    # --- Data Preparation ---
    # The 'moment' is the molecular property we want to predict.
    # It is a vector of 3 values. We will only use the first moment for this example.
    first_moment_value = [raw_data['moment'][key][0] for key in raw_data['moment']]

    # Normalize the target values to be between 0 and 1.
    # This helps with model training.
    first_moment_list = np.array(first_moment_value)
    first_moment_list -= np.min(first_moment_list)
    first_moment_list /= np.max(first_moment_list)

    # Create a TensorFlow dataset from the data generator.
    data = tf.data.Dataset.from_generator(
        lambda: data_generator(raw_data, first_moment_list),
        output_types=((tf.float32, tf.float32), tf.float32),
        output_shapes=(
            (tf.TensorShape([None, N_FEATURES]), tf.TensorShape([None, None])),
            tf.TensorShape([]),
        ),
    )

    # Split the data into training, validation, and test sets.
    # The dataset is shuffled before splitting.
    data = data.shuffle(1000)
    test_data = data.take(1928)
    val_data = data.skip(1928).take(1928)
    train_data = data.skip(3856)

    # --- Model Training ---
    model = build_model()
    model.compile("adam", loss="mean_squared_error")

    # Train the model.
    result = model.fit(train_data.batch(1), validation_data=val_data.batch(1), epochs=10)

    # --- Save the Model ---
    if not os.path.exists('model'):
        os.makedirs('model')
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # --- Evaluation and Plotting ---
    # Plot the training and validation loss.
    plt.plot(result.history["loss"], label="training")
    plt.plot(result.history["val_loss"], label="validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.show()

    # Evaluate the model on the test data.
    yhat = model.predict(test_data.batch(1), verbose=0)[:, 0]
    test_y = [y for x, y in test_data]

    # Plot the predicted values against the true values.
    plt.figure()
    plt.plot(test_y, test_y, "-", label="Ideal")
    plt.plot(test_y, yhat, ".", label="Predicted")
    plt.legend()
    plt.title("Testing Data: Predicted vs. True Values")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    # Add correlation and loss metrics to the plot.
    plt.text(
        min(test_y) + 0.1,
        max(test_y) - 0.2,
        f"correlation = {np.corrcoef(test_y, yhat)[0,1]:.3f}",
    )
    plt.text(
        min(test_y) + 0.1,
        max(test_y) - 0.3,
        f"loss = {np.sqrt(np.mean((np.array(test_y) - np.array(yhat))**2)):.3f}",
    )
    plt.show()

if __name__ == "__main__":
    main()
