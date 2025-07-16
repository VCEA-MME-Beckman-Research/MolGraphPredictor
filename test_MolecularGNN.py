import unittest
import numpy as np
from MolecularGNN import gen_smiles2graph, N_FEATURES

class TestMolecularGNN(unittest.TestCase):

    def test_gen_smiles2graph(self):
        """Test the SMILES to graph conversion for a simple molecule (methane)."""
        # Methane (CH4)
        sml = "C"
        nodes, adj = gen_smiles2graph(sml)

        # Expected number of atoms in methane (1 Carbon, 4 Hydrogens)
        self.assertEqual(nodes.shape[0], 5)
        # Expected number of features for each atom
        self.assertEqual(nodes.shape[1], N_FEATURES)
        # Expected shape of the adjacency matrix
        self.assertEqual(adj.shape, (5, 5))

        # Check if the adjacency matrix is symmetric
        self.assertTrue(np.allclose(adj, adj.T))

        # Check the diagonal of the adjacency matrix (should be all 1s due to self-loops)
        self.assertTrue(np.all(np.diag(adj) == 1))

if __name__ == '__main__':
    unittest.main()
