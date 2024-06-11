import unittest
from qBN.qBNclass import qBayesNet
import pyAgrum as gum

class TestQInfBN(unittest.TestCase):
    """
    Unit tests for the qInfBN class, focusing on verifying the correctness of quantum inference operations.

    Attributes
    ----------
    bn : qBayesNet
        The qBayesNet instance used to create the qInfBN instance for testing.
    qinf : qInfBN
        The qInfBN instance under test which utilizes quantum circuits for Bayesian inference.

    Methods
    -------
    setUp(self):
        Sets up a qBayesNet and qInfBN instance for each test.

    test_getA(self):
        Tests the getA method to ensure it correctly constructs the quantum sample preparation Operator.

    test_addA(self):
        Ensures the addA method correctly integrates the quantum operator into an existing quantum circuit.

    test_getAdjoint(self):
        Validates that getAdjoint correctly computes the adjoint of a given quantum Operator.

    test_addInverse(self):
        Tests that addInverse correctly integrates the inverse of a given quantum circuit into an existing circuit.

    test_getB(self):
        Verifies that getB constructs the correct B gate based on provided evidence.

    test_addB(self):
        Ensures that addB correctly adds the B gate to a quantum circuit based on given evidence.

    test_getZ(self):
        Checks that getZ correctly constructs the Z gate for provided evidence.

    test_addZ(self):
        Validates that addZ correctly adds the Z gate to a quantum circuit based on given evidence.

    test_getS(self):
        Tests the getS method to ensure it correctly constructs the S gate as a phase flip operator.

    test_addS(self):
        Ensures that addS correctly integrates the S gate into a given quantum circuit.

    test_getG(self):
        Verifies the getG method correctly constructs the Grover iterate based on the gate A and evidence.

    test_addG(self):
        Checks that addG correctly adds the Grover iterate to a quantum circuit.
    """

    def setUp(self):
        """
        Initializes the qBayesNet and qInfBN instances before each test method.
        """
        self.bn = qBayesNet(gum.fastBN("A->B<-C", 2))
        self.qinf = qInfBN(self.bn)

    def test_getA(self):
        """
        Ensure that getA returns a valid Operator object representing the quantum circuit without measurement.
        """
        A = self.qinf.getA()
        self.assertIsInstance(A, Operator, "getA should return an Operator instance")

    def test_addA(self):
        """
        Ensure that addA properly appends the quantum circuit representation to an existing quantum circuit.
        """
        circuit = QuantumCircuit(3)
        self.qinf.addA(circuit)
        self.assertEqual(len(circuit.data), 1, "addA should add one element to the circuit")

    def test_getAdjoint(self):
        """
        Ensure that getAdjoint returns the correct adjoint of a given operator.
        """
        A = self.qinf.getA()
        A_adj = self.qinf.getAdjoint(A)
        self.assertEqual(A_adj.label, A.label + '†', "getAdjoint should return the adjoint label correctly")

    def test_addInverse(self):
        """
        Ensure that addInverse correctly adds the inverse of the operator to the circuit.
        """
        circuit = QuantumCircuit(3)
        A = self.qinf.getA().to_instruction()
        self.qinf.addInverse(circuit, A)
        self.assertEqual(len(circuit.data), 1, "addInverse should add one element to the circuit")

    def test_getB(self):
        """
        Test getB with specific evidence to ensure the phase flip operator is constructed correctly.
        """
        evidence = {0: 1}
        B = self.qinf.getB(evidence)
        self.assertIsInstance(B, Operator, "getB should return an Operator instance")

    def test_addB(self):
        """
        Verify that addB correctly adds the B gate to the quantum circuit.
        """
        circuit = QuantumCircuit(3)
        evidence = {0: 1}
        self.qinf.addB(circuit, evidence)
        self.assertEqual(len(circuit.data), 1, "addB should add one element to the circuit")

    def test_getZ(self):
        """
        Ensure getZ constructs the Z gate correctly based on evidence.
        """
        evidence = {0: 1}
        Z = self.qinf.getZ(evidence)
        self.assertIsInstance(Z, Operator, "getZ should return an Operator instance")

    def test_addZ(self):
        """
        Check that addZ correctly integrates the Z gate based on evidence.
        """
        circuit = QuantumCircuit(3)
        evidence = {0: 1}
        self.qinf.addZ(circuit, evidence)
        self.assertEqual(len(circuit.data), 1, "addZ should add one element to the circuit")

if __name__ == "__main__":
    unittest.main()
