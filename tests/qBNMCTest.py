import unittest
import os
import sys
sys.path.insert(0, os.path.abspath('../'))
from unittest.mock import patch, MagicMock
from pyAgrum import DiscreteVariable, Instantiation, BayesNet, LabelizedVariable, Potential
from qBN.qBNMC import qBNMC
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import RYGate, XGate
import numpy as np
from typing import Union #List and Dict are deprecated (python 3.9)
import pyAgrum as gum

class TestqBNMC(unittest.TestCase):
    
    def setUp(self):
        self.mock_bn = MagicMock(spec=BayesNet)
        self.mock_bn.nodes.return_value = {0, 1, 2}

        self.mock_variable0 = MagicMock(spec=DiscreteVariable)
        self.mock_variable0.domainSize.return_value = 4
        self.mock_variable0.name.return_value = "Variable0"

        self.mock_variable1 = MagicMock(spec=DiscreteVariable)
        self.mock_variable1.domainSize.return_value = 2
        self.mock_variable1.name.return_value = "Variable1"

        self.mock_variable2 = MagicMock(spec=DiscreteVariable)
        self.mock_variable2.domainSize.return_value = 2
        self.mock_variable2.name.return_value = "Variable2"

        self.mock_bn.variable.side_effect = lambda n_id: {
            0: self.mock_variable0,
            1: self.mock_variable1,
            2: self.mock_variable2
        }[n_id]

        self.qb_net = qBNMC(self.mock_bn)
        self.qb_net.n_qb_map = {0: [0, 1], 1: [2, 3], 2: [4, 5]}

        bif_path = os.path.abspath("tests/asia.bif")
        self.asia_bn = gum.loadBN(bif_path)
        self.qb_asia_net = qBNMC(self.asia_bn)

    def test_initialization(self):
        self.assertEqual(self.qb_net.bn, self.mock_bn)
        expected_map = {
            0: [0, 1],
            1: [2, 3],
            2: [4, 5]
        }
        with patch.object(qBNMC, 'getWidth', side_effect=[2, 2, 2]):
            self.qb_net = qBNMC(self.mock_bn)
            self.assertEqual(self.qb_net.n_qb_map, expected_map)

    def test_mapNodeToQBit(self):
        nodes = {0, 1, 2}
        with patch.object(qBNMC, 'getWidth', side_effect=[2, 2, 2]):
            qb_map = self.qb_net.mapNodeToQBit(nodes)
        
        expected_map = {
            0: [0, 1],
            1: [2, 3],
            2: [4, 5]
        }
        self.assertEqual(qb_map, expected_map)

    def test_getWidth(self):
        node = 0
        width = self.qb_net.getWidth(node)
        self.assertEqual(width, 2)

    def test_getBinarizedParameters(self):
        width_dict = {'A': 2, 'B': 3}
        param_dict = {'A': 2, 'B': 5}

        # Mock the nodeId method to return a consistent ID for each variable
        self.mock_bn.nodeId.side_effect = lambda var: ord(var.name()) - ord('A')

        # Mock the variable method to return a mock variable with a name matching the input key
        def variable_side_effect(key):
            mock_var = MagicMock()
            mock_var.name.return_value = key
            return mock_var

        self.mock_bn.variable.side_effect = variable_side_effect

        # Expected result based on the binary representation of 2 and 5
        expected_result = {
            0: [1, 0],  # Binary representation of 2 with width 2
            1: [1, 0, 1]  # Binary representation of 5 with width 3
        }

        result = self.qb_net.getBinarizedParameters(width_dict, param_dict)
        self.assertEqual(result, expected_result)

    def test_getRootNodes(self):
        self.mock_bn.nodes.return_value = {0, 1, 2, 3}
        self.mock_bn.parents.side_effect = lambda node_id: [] if node_id in {0, 2} else [0]

        expected_roots = {0, 2}
        result = self.qb_net.getRootNodes()
        self.assertEqual(result, expected_roots)

    def test_getAllParentSates(self):
        expected_output1 = [{'smoking': 0}, {'smoking': 1}]
        actual_output1 = self.qb_asia_net.getAllParentSates('lung_cancer')

        expected_output2 = [{'tuberculosis': 0, 'lung_cancer': 0},
                            {'tuberculosis': 1, 'lung_cancer': 0},
                            {'tuberculosis': 0, 'lung_cancer': 1},
                            {'tuberculosis': 1, 'lung_cancer': 1}]
        actual_output2 = self.qb_asia_net.getAllParentSates('tuberculos_or_cancer')

        expected_output3 = [{'tuberculos_or_cancer': 0, 'bronchitis': 0},
                            {'tuberculos_or_cancer': 1, 'bronchitis': 0},
                            {'tuberculos_or_cancer': 0, 'bronchitis': 1},
                            {'tuberculos_or_cancer': 1, 'bronchitis': 1}]
        actual_output3 = self.qb_asia_net.getAllParentSates('dyspnoea')

        self.assertEqual(actual_output1, expected_output1)
        self.assertEqual(actual_output2, expected_output2)
        self.assertEqual(actual_output3, expected_output3)
        
        
    def test_getQuantumRegisters(self):

        def variable_side_effect(n_id):
            var = LabelizedVariable(str(n_id), "label", ["0", "1"])  
            var.domainSize = MagicMock(return_value=2)
            return var

        self.mock_bn.variable.side_effect = variable_side_effect


        expected_result = {
            0: QuantumRegister(int(np.ceil(np.log2(2))), name='0'),  
            1: QuantumRegister(int(np.ceil(np.log2(2))), name='1'),  
            2: QuantumRegister(int(np.ceil(np.log2(2))), name='2')   
        }

        result = self.qb_net.getQuantumRegisters()
        
        
        self.assertEqual(len(result), len(expected_result))
        for key in result:
            self.assertTrue(key in expected_result)
            self.assertEqual(result[key].size, expected_result[key].size)
            self.assertEqual(result[key].name, expected_result[key].name)

    def test_indicatorFunction(self):

        binary_list = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
        targets = {2: 1}
        expected_result = [True, True, True, True]
        result = self.qb_net.indicatorFunction(binary_list, targets)
        self.assertEqual(result, expected_result)


        binary_list = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]
        targets = {2: 1}
        expected_result = [False, False, False, False]
        result = self.qb_net.indicatorFunction(binary_list, targets)
        self.assertEqual(result, expected_result)


        binary_list = [[0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0]]
        targets = {1: 1, 2: 0}
        expected_result = [False, True, False, True]
        result = self.qb_net.indicatorFunction(binary_list, targets)
        self.assertEqual(result, expected_result)


        binary_list = [[0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0]]
        targets = {}
        expected_result = [False, False, False, False]
        result = self.qb_net.indicatorFunction(binary_list, targets)
        self.assertEqual(result, expected_result)

    @patch.object(qBNMC, 'getWidth', return_value=2)
    @patch.object(qBNMC, 'indicatorFunction', return_value=[True, False, True, False])
    def test_getProbability(self, mock_indicatorFunction, mock_getWidth):

        mock_cpt = MagicMock()
        mock_cpt.__getitem__.side_effect = lambda x: np.array([0.1, 0.2, 0.3, 0.4]) if x == {0: 1} else np.array([0.0, 0.0, 0.0, 0.0])
        self.mock_bn.cpt.return_value = mock_cpt


        value = 1
        node = 0
        qb_id = 0
        param_qbs = {1: 0}
        param_nodes = {0: 1}

        # Expected result
        expected_result = 0.4  # Sum of probabilities where the indicator is True

        # Call the method
        result = self.qb_net.getProbability(value, node, qb_id, param_qbs, param_nodes)

        # Assertions
        self.assertEqual(result, expected_result)
        mock_getWidth.assert_called_once_with(node)
        mock_indicatorFunction.assert_called_once()

    @patch('qBN.qBNMC.transpile')
    @patch('qBN.qBNMC.SamplerV2')
    @patch('qBN.qBNMC.AerSimulator')
    def test_aerSimulation(self, MockAerSimulator, MockSamplerV2, mock_transpile):

        mock_backend = MagicMock()
        MockAerSimulator.return_value = mock_backend

        mock_sampler = MagicMock()
        MockSamplerV2.return_value = mock_sampler


        mock_transpiled_circuit = MagicMock(spec=QuantumCircuit)
        mock_transpile.return_value = mock_transpiled_circuit


        mock_job = MagicMock()
        mock_sampler.run.return_value = mock_job
        
        mock_result = MagicMock()
        mock_job.result.return_value = mock_result
        

        mock_result[0].data.meas.get_counts.return_value = {
            '000000': 5000,
            '111111': 5000,
        }


        qc = QuantumCircuit(6)
        qc.h([0, 1, 2, 3, 4, 5])


        res = self.qb_net.aerSimulation(qc, shots=10000)


        self.assertIsInstance(res, dict)
        self.assertIn("Variable0", res)
        self.assertEqual(len(res["Variable0"]), self.mock_variable0.domainSize())
        self.assertAlmostEqual(sum(res["Variable0"]), 1.0)



if __name__ == '__main__':
    unittest.main()
