import unittest
from unittest.mock import patch, MagicMock
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.quantum_info import Operator
from qBNMC import qBayesNet
from qBNRejection import qInference

class TestQInference(unittest.TestCase):

    def setUp(self):
        # Create a mock qBayesNet
        self.mock_qbn = MagicMock(spec=qBayesNet)
        
        # Mock getQuantumRegisters to return a dictionary of QuantumRegisters
        self.mock_qbn.getQuantumRegisters.return_value = {
            0: QuantumRegister(2, 'q0'),
            1: QuantumRegister(2, 'q1'),
            2: QuantumRegister(2, 'q2')
        }
        
        # Mock n_qb_map
        self.mock_qbn.n_qb_map = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
        
        # Initialize qInference with the mocked qBayesNet
        self.qinf = qInference(self.mock_qbn)

    def test_initialization(self):
        # Check if qbn is set correctly
        self.assertEqual(self.qinf.qbn, self.mock_qbn)
        
        # Check if q_registers are set correctly
        self.assertEqual(self.qinf.q_registers, self.mock_qbn.getQuantumRegisters())
        
        # Check if all_qbits are set correctly
        self.assertEqual(self.qinf.all_qbits, [0, 1, 2, 3, 4, 5])
        
        # Check initial values of other attributes
        self.assertEqual(self.qinf.evidence, dict())
        self.assertIsNone(self.qinf.inference_res)
        self.assertEqual(self.qinf.max_iter, 1000)
        self.assertEqual(self.qinf.log, {"A": 0, "B": 0})
        self.assertIsNone(self.qinf.A)
        self.assertIsNone(self.qinf.G)

    @patch('qBN.qBNMC.qBayesNet.buildCircuit')
    def test_getA(self, mock_buildCircuit):
        # Create a mock QuantumCircuit
        mock_circuit = MagicMock(spec=QuantumCircuit)
        
        # Mock the buildCircuit method to return the mock QuantumCircuit
        mock_buildCircuit.return_value = mock_circuit
        
        # Decompose the circuit (this is part of getA method)
        mock_decomposed_circuit = MagicMock(spec=QuantumCircuit)
        mock_circuit.decompose.return_value = mock_decomposed_circuit
        
        # Create an Operator from the decomposed circuit
        mock_operator = MagicMock(spec=Operator)
        with patch('qiskit.quantum_info.Operator', return_value=mock_operator):
            result = self.qinf.getA()
        
        # Assertions
        mock_buildCircuit.assert_called_once_with(add_measure=False)
        mock_circuit.decompose.assert_called_once()
        self.assertEqual(result, mock_operator.to_instruction())
        self.assertEqual(result.label, 'A')

if __name__ == '__main__':
    unittest.main()
