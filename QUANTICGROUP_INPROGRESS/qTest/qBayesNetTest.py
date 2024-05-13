import sys
import unittest
import numpy as np
from unittest.mock import patch, MagicMock, call
import pyAgrum as gum
from qBN.qBNclass import qBayesNet

class TestQBayesNet(unittest.TestCase):
    """
    Class used to test the qBayesNet Class
    
    Tests the creation and manipulation, mapping of nodes to quantum bits, quantum circuit construction, and probability calculations based on quantum states.
    
    Attributes
    ----------
    bn : BayesNet
        pyAgrum Bayesian Network initialized in the setup to test interactions with qBayesNet.
    qbn : qBayesNet
        The qBayesNet instance under test which integrates quantum circuit capabilities with Bayesian Network.

    Methods
    -------
    setUp(self):
        Initializes the Bayesian Network and qBayesNet instance for each test method.

    test_getWidth(self):
        Ensures that the getWidth method correctly computes the number of qubits required to represent a node.

    test_getRootNodes(self):
        Tests whether the getRootNodes method accurately identifies root nodes of the Bayesian Network.

    test_mapNodeToQBit(self):
        Verifies that the mapNodeToQBit method correctly maps network nodes to their corresponding qubits.

    test_indicatorFunction(self):
        Assesses the indicatorFunction method's accuracy in evaluating binary conditions.

    test_buildCircuit(self):
        Evaluates the buildCircuit method's ability to construct the appropriate quantum circuit based on the Bayesian Network.

    test_getProbability(self):
        Tests getProbability method to ensure it computes the correct probability of a qubit's state.

    test_runSimulation(self):
        Checks the complete simulation process, including circuit building, execution, and probability computation.
    """
    def setUp(self):
        """
        Sets up the test environment for each test method by initializing the Bayesian Network and the 
        qBayesNet instance.
        
        This method is automatically called before each test method to configure the necessary preconditions 
        and settings for the tests. 
        
        It initializes a Bayesian Network with a predefined structure and associates it with a 
        qBayesNet instance to be used in the tests.
        """
        self.bn = gum.fastBN("A->B<-C",2)
        self.net = qBayesNet(self.bn)
        
    @patch('qBN.qBNclass.qBayesNet.getWidth', return_value=1)
    def test_mapNodeToQBit(self, mock_get_width):
        """
        Verifies that the mapNodeToQBit method correctly maps the nodes of the Bayesian Network to 
        specific qubits in a quantum circuit.
        
        This test confirms that each node in the Bayesian Network is associated with the correct number and 
        set of qubits, ensuring that 
        the quantum circuit accurately represents the network structure.
        """        
        nodes = {1, 2, 3}
        expected = {
            1: [0],
            2: [1],
            3: [2]
        }
        result = self.net.mapNodeToQBit(nodes)
        self.assertEqual(result, expected)

    def test_getWidth(self):
        """
        Verifies the getWidth method to ensure it correctly calculates the number of qubits needed 
        based on the domain size of the node.
        """
        # Setup
        node_id = "A"
        domain_size = 4  # 2^2 requires 2 qubits
        expected_width = 2

        mock_variable = MagicMock()
        mock_variable.domainSize.return_value = domain_size
        self.net.bn.variable = MagicMock(return_value=mock_variable)

        width = self.net.getWidth(node_id)
        self.assertEqual(width, expected_width)
        mock_variable.domainSize.assert_called_once()  # Ensures that domainSize was called
        self.net.bn.variable.assert_called_with(node_id)  # Ensures that the correct node ID was used
    
    @patch('qBN.qBNclass.qBayesNet.getWidth')
    def test_getTotNumQBits(self, mock_get_width):
        """
        Verifies the getTotNumQBits method to ensure it correctly calculates the total number of qubits 
        required for all nodes in the Bayesian Network.
        """
        # Setup the mock to return specific widths for nodes
        node_ids = [self.bn.idFromName(name) for name in ['A', 'B', 'C']]  
        mock_get_width = patch('qBN.qBNclass.qBayesNet.getWidth', side_effect=[2, 3, 1]).start()
        
        # Expected total width is the sum of the individual widths
        expected_total_width = 2 + 3 + 1
    
        # Test
        total_width = self.net.getTotNumQBits()
        self.assertEqual(total_width, expected_total_width)
    
        # Verify that getWidth was called for each node in the network
        calls = [call(id) for id in node_ids]
        mock_get_width.assert_has_calls(calls, any_order=True)
        self.assertEqual(mock_get_width.call_count, len(node_ids))
        
        mock_get_width.stop()  # Stop the patch to avoid side effects on other tests

    def test_getBinarizedParameters(self):
        """
        Tests the getBinarizedParameters method to ensure it returns the correct binary representation 
        of variables' states.
        """
        # Setup input dictionaries
        width_dict = {'A': 2, 'B': 1, 'C': 3}  # Widths for A, B, C
        param_dict = {'A': 2, 'B': 0, 'C': 5}  # States for A, B, C

        # Expected binary representations (as lists of integers)
        expected_bin_params = {
            0 : [1, 0],
            1 : [0],
            2 : [1, 0, 1]
        }

        qbn=self.net
        
        # Execute the method
        bin_params = qbn.getBinarizedParameters(width_dict, param_dict)

        # Test the output
        self.assertEqual(bin_params, expected_bin_params)

if __name__ == "__main__":
    unittest.main()
