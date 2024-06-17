from .qBNMC import qBayesNet #relative import

import numpy as np

from typing import Union #List and Dict are deprecated (python 3.9)

from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit.library import XGate, ZGate
from qiskit.quantum_info import Operator

import scipy.linalg #for qiskit transpile function

from pyAgrum import Potential, BayesNetFragment

class qInference:
    """
    Class used to perform inference via rejection sampling from a Quantum Circuit
    representation of a Bayesian Network
    Based on the paper:
    Quantum Inference on Bayesian Networks - Guang Hao Low

    Attributes
    ----------

    qbn: qBayesNet
        Quantum Baysien Network object

    q_registers: dict[int: QuantumRegister]
        Quantum Registers used to build the rotation gates

    all_qbits: list[int]
        List containing all qubit IDs

    evidence: dict[Union[str, int]: int]
        Dictionary with variable names or IDs as keys and their state as values

    inference_res: dict[Union[str, int]: list[float]]
        Dictionary with variable names or IDs as keys and their inferred CPTs as values

    max_iter: int
        Number of iterations (samples)

    log: dict[str, int]
        Dictionary with gate names as keys and their number of usage in last run as values

    A: QuantumCircuit
        Quantum Circuit of q-sample preparator

    G: QuantumCircuit
        Quantum Circuit of Grover's iteration

    Methods
    -------

    maxIter(self) -> None:
        Returns the maximum iteration number

    addA(self, circuit: QuantumCircuit) -> None:
        Composes the quantum sample preparation Operator object
        Operator of the Quantum Circuit representing the Baysian Network to circuit

    addInverse(self, circuit: QuantumCircuit, M: QuantumCircuit) -> None:
        Composes the adjoint operator of M to circuit

    addB(self, circuit: QuantumCircuit, evidence_qbs: dict[int, int]) -> Operator:
        Composed the B gate of the phase flip operator to circuit (eq7)

    addZ(self, circuit: QuantumCircuit, evidence_qbs: dict[int, int]) -> None:
        Composes the Z gate of the phase flip operator to circuit (eq7)

    addS(self, circuit: QuantumCircuit, evidence_qbs: dict[int, int]) -> None:
        Composes the phase flip operator to circuit (eq7)

    addG(self, circuit: QuantumCircuit, A: QuantumCircuit, 
               evidence_qbs: dict[int, int]) -> None:
        Composes the grover iterate to circuit

    getGates(self) -> None:
        Stores A gate and G gate

    transpileGates(self) -> None:
        Transpiles saved A and G gates with optimization level set to 3

    getEvidenceQuBits(self, evidence: dict[int: int]) -> dict[int, int]:
        Gives qubit representation of evidence in Baysian Network

    getSample(self, A: Operator, G: Operator,
                        evidence: dict[Union[str, int]: int],
                        verbose: int = 0) -> dict[int: int]:
        Generate one sample from evidence (Algorithm 1)

    makeInference(self, verbose: int = 0) -> dict[Union[str, int]: list[float]]:
        Performs rejection sampling on Quantum Circuit representation of
        Baysian Network

    setEvidence(self, evidence: dict[Union[str, int]: int]) -> None:
        Sets the evidence of the rejection sampler

    setMaxIter(self, max_iter: int = 1000) -> None:
        Sets the max iteration of the rejection sampler

    posterior(self, node: Union[str, int]) -> Potential:
        Gives the probability table of the node variable from sampling results

    useFragmentBN(self, evidence: set[Union[str, int]] = None, 
                            target: set[Union[str, int]] = None) -> None:
        Uses fragmented Baysian Network to speed up circuit computations

    """

    def __init__(self, qbn: qBayesNet) -> None:
        """
        Initialises the qInference Object 

        Parameters
        ----------
        qbn : qBayesNet
            Quantum Bayesian Network

        """

        self.qbn = qbn
        self.q_registers = self.qbn.getQuantumRegisters()
        self.all_qbits = np.hstack(list(self.qbn.n_qb_map.values())).tolist()
        self.evidence = dict()
        self.inference_res = None
        self.max_iter = 1000
        self.log = {"A": 0, "B": 0}
        self.A = None
        self.G = None

    def maxIter(self) -> None:
        """
        Returns the maximum iteration number

        Returns
        -------
        int
            Maximum iteration number

        """

        return self.max_iter

    def addA(self, circuit: QuantumCircuit) -> None:
        """
        Composes the quantum sample preparation Operator object
        Operator of the Quantum Circuit representing the Baysian Network to circuit

        Returns
        -------
        Operator
            Quantum gate A

        """

        A = self.qbn.buildCircuit(add_measure=False)
        circuit.compose(A, inplace=True)
        circuit.barrier()
        return
    
    def addInverse(self, circuit: QuantumCircuit, M: QuantumCircuit) -> None:
        """
        Composes the adjoint operator of M to circuit

        Parameters
        ----------
        M: Operator
            Operator to be transformed

        Returns
        -------
        Operator
            M adjoint

        """

        circuit.compose(M.inverse(), inplace=True)
        return

    def addB(self, circuit: QuantumCircuit, evidence_qbs: dict[int, int]) -> None:
        """
        Composed the B gate of the phase flip operator to circuit (eq7)

        Parameters
        ----------
        evidence_qbs: dict[int, int]
            Dictionary with qubit IDs as keys and their quantum state as values

        Returns
        -------
        Operator
            Quantum gate B

        """

        for qb_id, qb_state in evidence_qbs.items():
            if qb_state == 0:
                circuit.compose(XGate(), [qb_id], inplace=True)

    def addZ(self, circuit: QuantumCircuit, evidence_qbs: dict[int, int]) -> None:
        """
        Composes the Z gate of the phase flip operator to circuit (eq7)

        Parameters
        ----------
        evidence_qbs: dict[int, int]
            Dictionary with qubit IDs as keys and their quantum state as values

        Returns
        -------
        Operator
            Quantum gate Z

        """

        rotation = ZGate()

        if len(evidence_qbs) > 1:
            rotation = rotation.control(len(evidence_qbs) - 1)

        circuit.compose(rotation, list(evidence_qbs.keys()), inplace=True)
        return

    def addS(self, circuit: QuantumCircuit, evidence_qbs: dict[int, int]) -> None:
        """
        Composes the phase flip operator to circuit (eq7)

        Parameters
        ----------
        evidence_qbs: dict[int, int]
            Dictionary with qubit IDs as keys and their quantum state as values

        Returns
        -------
        Operator
            Quantum gate S

        """

        self.addB(circuit, evidence_qbs)
        self.addZ(circuit, evidence_qbs)
        self.addB(circuit, evidence_qbs)
        return

    def addG(self, circuit: QuantumCircuit, A: QuantumCircuit, 
                   evidence_qbs: dict[int, int]) -> None:
        """
        Composes the grover iterate to circuit

        Parameters
        ----------
        A: Operator
            Gate A
        evidence_qbs: dict[int, int]
            Dictionary with qubit IDs as keys and their quantum state as values

        Returns
        -------
        Operator
            Quantum gate G

        """

        res = None

        all_qbits = np.ravel(list(self.qbn.n_qb_map.values())).tolist()

        res = self.addS(circuit, evidence_qbs)
        circuit.barrier(label='S\u2091')
        res = self.addInverse(circuit, A)
        circuit.barrier(label='A\u207B\u00B9')
        res = self.addS(circuit, {qb_id: 0 for qb_id in all_qbits})
        circuit.barrier(label='S\u2080')
        res = circuit.compose(A, inplace=True)
        circuit.barrier(label='A')

        return
    
    def getGates(self) -> None:
        """
        Stores A gate and G gate

        """

        evidence_n_id = {self.qbn.bn.nodeId(self.qbn.bn.variable(key)): val
                         for key, val in self.evidence.items()}
        evidence_qbs = self.getEvidenceQuBits(evidence_n_id)

        self.A  = QuantumCircuit(*list(self.q_registers.values()))
        self.addA(self.A)

        self.G = QuantumCircuit(*list(self.q_registers.values()))
        self.addG(self.G, self.A, evidence_qbs, inplace=False)

    def transpileGates(self) -> None:
        """
        Transpiles saved A and G gates with optimization level set to 3

        """
    
        self.A = transpile(self.A, optimization_level=3)
        self.G = transpile(self.G, optimization_level=3)

    def getEvidenceQuBits(self, evidence: dict[int: int]) -> dict[int, int]:
        """
        Gives qubit representation of evidence in Baysian Network

        Parameters
        ----------
        evidence: dict[Union[str, int]: int]
            Dictionary with variable IDs as keys and their state as values

        Returns
        -------
        dict[int, int]
            Dictionary with qubit IDs as keys and their quantum state as values

        """

        res = dict()

        for n_id, n_state in evidence.items():
            bin_state = np.binary_repr(n_state, width=self.qbn.getWidth(n_id))
            for qb_num in range(len(bin_state)):
                res[self.qbn.n_qb_map[n_id][qb_num]] = int(bin_state[qb_num])

        return res

    def getSample(self, A: Operator, G: Operator,
                        evidence: dict[Union[str, int]: int],
                        verbose: int = 0) -> dict[int: int]:
        """
        Generate one sample from evidence (Algorithm 1)

        Parameters
        ----------
        A: Operator
            Gate A
        G: Operator
            Gate G
        evidence: dict[Union[str, int]: int]
            Dictionary with variable IDs as keys and their state as values
        verbose: int = 0
            Verbose of function

        Returns
        -------
        dict[int, int]
            Dictionary with variable IDs as keys and their state as values
        
        """

        cl_reg = ClassicalRegister(len(self.all_qbits), "meas")
        circuit = QuantumCircuit(*list(self.q_registers.values()), cl_reg)
        circuit.compose(A, inplace=True)

        circuit.measure_all(add_bits=False)

        k = -2

        while True:

            k = k + 1

            for i in range(1, A.num_qubits+1):
                p = circuit.data.pop(-1)

            circuit.compose(G.power(int(np.ceil(2.0**k))), inplace=True)

            circuit.measure_all(add_bits=False)

            run_res = self.qbn.aerSimulation(circuit, shots=1)
            self.log['A'] += 1
            self.log['G'] += 2**(k+1)


            run_res = {self.qbn.bn.nodeId(self.qbn.bn.variable(node)): \
                       state.index(1.0) for node, state in run_res.items()}

            if verbose > 0:
                print(f"k = {k+1}, A: +1, G: +{2**(k+1)}")
                print(f"run_res = {run_res}")

            match_evidence = True

            for node, state in evidence.items():
                n_id = self.qbn.bn.nodeId(self.qbn.bn.variable(node))
                if verbose > 0:
                    print(f"node = {node}, \
                            state = {state}, \
                            run_res[node] = {run_res[n_id]}")
                if run_res[n_id] != state:
                    match_evidence = False
                    break

            if match_evidence:
                break

        return run_res

    def makeInference(self, verbose: int = 0) -> dict[Union[str, int]: list[float]]:
        """
        Performs rejection sampling on Quantum Circuit representation of
        Baysian Network

        Parameters
        ----------
        verbose: int = 0
            Verbose of function

        Returns
        -------
        dict[Union[str, int]: list[float]]
            Dictionary with variable names as keys and proability vector as values

        """

        if self.A is None or self.G is None: self.getGates()

        self.log = {"A": 0, "G": 0}

        res = {node: [0] * self.qbn.bn.variable(node).domainSize()
               for node in self.qbn.n_qb_map.keys()}

        for i in range(self.max_iter):
            sample = self.getSample(self.A, self.G, self.evidence, 
                                    verbose = 1 if verbose == 2 else 0)

            for node, state in sample.items():
                res[node][state] += 1.0/self.max_iter

            if verbose == 1: print(f"iteration: {i}, log: {self.log}")

        res = {self.qbn.bn.variable(key).name(): val for key, val in res.items()}

        self.inference_res = res

        return res

    def setEvidence(self, evidence: dict[Union[str, int]: int]) -> None:
        """
        Sets the evidence of the rejection sampler

        Parameters
        ----------
        evidence: dict[Union[str, int]: int]
            Dictionary with variable IDs as keys and their state as values

        """
        self.evidence = evidence

    def setMaxIter(self, max_iter: int = 1000) -> None:
        """
        Sets the max iteration of the rejection sampler

        Parameters
        ----------
        max_iter: int = 1000
            Max iteration

        """
        self.max_iter = max_iter

    def posterior(self, node: Union[str, int]) -> Potential:
        """
        Gives the probability table of the node variable from sampling results

        Parameters
        ----------
        node: Union[str, int]
            Variable name or ID

        Returns
        -------
        Potential
            pyAgrum.Potential object

        """

        name = self.qbn.bn.variable(node).name()
        potential = Potential().add(self.qbn.bn.variable(name))

        if self.inference_res is None: self.makeInference()
        potential.fillWith(self.inference_res[name])

        return potential

    def useFragmentBN(self, evidence: set[Union[str, int]] = None, 
                            target: set[Union[str, int]] = None) -> None:
        """
        Uses fragmented Baysian Network to speed up circuit computations

        Parameters
        ----------
        evidence: set[Union[str, int]] = None
            Evidence variables to be added to the ones already stored in object
        target: set[Union[str, int]] = None
            Target varaible (default is set to empty set)

        """

        if evidence is None: evidence = set()
        if target is None: target = set()
        evidence = evidence.union(self.evidence.keys())

        fbn = BayesNetFragment(self.qbn.bn)

        for node in target.union(evidence):
            fbn.installAscendants(node)

        self.qbn = qBayesNet(fbn.toBN())
        self.q_registers = self.qbn.getQuantumRegisters()
        self.all_qbits = np.hstack(list(self.qbn.n_qb_map.values())).tolist()
