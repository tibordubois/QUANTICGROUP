from qBN.qBNclass import qBayesNet

import numpy as np

from typing import Union #List and Dict are deprecated (python 3.9)

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import XGate, ZGate
from qiskit.quantum_info import Operator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import StatevectorSampler

from qiskit.visualization import array_to_latex

class qInfBN:
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

    Methods
    -------
    getA(self) -> Operator:
        Gives the Operator object represenation of the Quantum Circuit representing
        the Baysian Network
    """

    def __init__(self, qbn: qBayesNet) -> None:
        """
        Initializes the quantum inference Bayesian Network instance with a specified qBayesNet object.

        Parameters
        ----------
        qbn : qBayesNet
            Quantum Bayesian Network instance to be used for inference
        """

        self.qbn = qbn
        self.q_registers = self.qbn.getQuantumRegisters()


    def getA(self) -> Operator:
        """
        Constructs and returns the quantum sample preparation Operator object from the Bayesian Network's quantum circuit.

        Returns
        -------
        Operator
            The Operator object representing the quantum gate A, without measurement
        """

        circuit = self.qbn.buildCircuit(add_measure=False)
        A = Operator(circuit)
        A = A.to_instruction()
        A.label = 'A'
        return A
    
    def addA(self, circuit: QuantumCircuit):
        """
        Adds the quantum sample preparation Operator from the Bayesian Network's quantum circuit into an existing quantum circuit.

        Parameters
        ----------
        circuit : QuantumCircuit
            The quantum circuit to which the A operator will be added
        """

        A = self.qbn.buildCircuit(add_measure=False)
        circuit.compose(A, inplace=True)
        circuit.barrier()
        return

    def getAdjoint(self, M: Operator):
        """
        Constructs and returns the adjoint (inverse) of the given operator M.

        Parameters
        ----------
        M : Operator
            The operator for which the adjoint is required

        Returns
        -------
        Operator
            The adjoint of operator M
        """

        M_label = M.label
        M = Operator(M.adjoint())
        M = M.to_instruction()
        M.label = M_label+'\u2020'
        return M
    
    def addInverse(self, circuit: QuantumCircuit, M: QuantumCircuit):
        """
        Adds the inverse of operator M into the given quantum circuit.

        Parameters
        ----------
        circuit : QuantumCircuit
            The quantum circuit to which the inverse of M will be added
        M : QuantumCircuit
            The quantum circuit representing the operator to be inverted
        """
        circuit.compose(M.inverse(), inplace=True)
        return

    def getB(self, evidence_qbs: dict[int, int]) -> Operator:
        """
        Constructs and returns the B gate, a phase flip operator, based on the provided evidence.

        Parameters
        ----------
        evidence_qbs : dict[int, int]
            Dictionary mapping qubit IDs to their corresponding quantum states

        Returns
        -------
        Operator
            The quantum gate B as an Operator object
        """

        circuit = QuantumCircuit(*list(self.q_registers.values()))

        for qb_id, qb_state in evidence_qbs.items():
            if qb_state == 0:
                circuit.append(XGate(), [qb_id])

        B = Operator(circuit)
        B = B.to_instruction()
        B.label = 'B'
        return B
    
    def addB(self, circuit: QuantumCircuit, evidence_qbs: dict[int, int]) -> Operator:
        """
        Adds the B gate, a phase flip operator, to the specified quantum circuit based on the given evidence.

        Parameters
        ----------
        circuit : QuantumCircuit
            The quantum circuit to which the B gate will be added
        evidence_qbs : dict[int, int]
            Dictionary mapping qubit IDs to their corresponding quantum states
        """

        for qb_id, qb_state in evidence_qbs.items():
            if qb_state == 0:
                circuit.compose(XGate(), [qb_id], inplace=True)

    def getZ(self, evidence_qbs: dict[int, int]) -> Operator:
        """
        Constructs and returns the Z gate, a controlled phase flip operator, for the given evidence.

        Parameters
        ----------
        evidence_qbs : dict[int, int]
            Dictionary mapping qubit IDs to their corresponding quantum states

        Returns
        -------
        Operator
            The quantum gate Z as an Operator object
        """

        circuit = QuantumCircuit(*list(self.q_registers.values()))

        rotation = ZGate()
        if len(evidence_qbs) > 1:
            rotation = rotation.control(len(evidence_qbs) - 1)

        circuit.append(rotation, list(evidence_qbs.keys()))

        #print(circuit)

        Z = Operator(circuit)
        Z = Z.to_instruction()
        Z.label = 'Z'
        return Z
    
    def addZ(self, circuit: QuantumCircuit, evidence_qbs: dict[int, int]) -> Operator:
        """
        Adds the Z gate of the phase flip operator (eq7)

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

    def getS(self, evidence_qbs: dict[int, int]) -> Operator:
        """Gives the phase flip operator (eq7)

        Parameters
        ----------
        evidence_qbs: dict[int, int]
            Dictionary with qubit IDs as keys and their quantum state as values

        Returns
        -------
        Operator
            Quantum gate S
        """

        circuit = QuantumCircuit(*list(self.q_registers.values()))

        all_qbits = np.ravel(list(self.qbn.n_qb_map.values())).tolist()

        B = self.getB(evidence_qbs)
        Z = self.getZ(evidence_qbs)

        circuit.append(B, qargs=all_qbits)
        circuit.append(Z, qargs=all_qbits)
        circuit.append(B, qargs=all_qbits)

        evidence_string = ''.join([str(q_state) for q_state in evidence_qbs.values()])

        S = Operator(circuit)
        S = S.to_instruction()
        label = 'S'+evidence_string

        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        label = label.translate(SUB)

        S.label = label

        return S
    
    def addS(self, circuit: QuantumCircuit, evidence_qbs: dict[int, int]) -> Operator:
        """
        Adds the phase flip operator (denoted as S) to the given quantum circuit.
    
        Parameters
        ----------
        circuit : QuantumCircuit
            The quantum circuit to which the phase flip operator will be added.
        evidence_qbs : dict[int, int]
            Dictionary with qubit IDs as keys and their quantum state as values.
    
        Returns
        -------
        None
        """
        self.addB(circuit, evidence_qbs)
        self.addZ(circuit, evidence_qbs)
        self.addB(circuit, evidence_qbs)
        return

    def getG(self, A: Operator, evidence_qbs: dict[int, int]) -> Operator:
        """
        Generates the Grover iterate operator based on gate A and the evidence provided.
    
        Parameters
        ----------
        A : Operator
            The quantum gate A (Amplitude amplification gate).
        evidence_qbs : dict[int, int]
            Dictionary with qubit IDs as keys and their quantum state as values.
    
        Returns
        -------
        Operator
            Quantum gate G (Grover iterate).
        """

        circuit = QuantumCircuit(*list(self.q_registers.values()))

        all_qbits = np.ravel(list(self.qbn.n_qb_map.values())).tolist()

        Se = self.getS(evidence_qbs)
        S0 = self.getS({qb_id: 0 for qb_id in all_qbits})

        A_adj = self.getAdjoint(A)

        circuit.append(Se, qargs=all_qbits)
        circuit.append(A_adj, qargs=all_qbits)
        circuit.append(S0, qargs=all_qbits)
        circuit.append(A, qargs=all_qbits)

        #print(circuit)

        G = Operator(circuit)
        G = G.to_instruction()
        G.label = 'G'

        return G

    def addG(self, circuit: QuantumCircuit, 
                   A: QuantumCircuit, 
                   evidence_qbs: dict[int, int], inplace: bool = True) -> Operator:
        """
        Adds the Grover iterate to the specified quantum circuit.
    
        Parameters
        ----------
        circuit : QuantumCircuit
            The quantum circuit to which the Grover iterate will be added.
        A : Operator
            The quantum gate A used in the Grover iterate.
        evidence_qbs : dict[int, int]
            Dictionary with qubit IDs as keys and their quantum state as values.
    
        Returns
        -------
        None
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

        return res

    def getEvidenceQuBits(self, evidence: dict[int: int]) -> dict[int, int]:
        """
        Translates a dictionary of evidence into qubit representations suitable for quantum computing.
    
        Parameters
        ----------
        evidence : dict[int, int]
            Dictionary with variable IDs as keys and their state as values.
    
        Returns
        -------
        dict[int, int]
            Dictionary with qubit IDs as keys and their quantum state as values.
        """
        res = dict()

        for n_id, n_state in evidence.items():
            bin_state = np.binary_repr(n_state, width=self.qbn.getWidth(n_id))
            for qb_num in range(len(bin_state)):
                res[self.qbn.n_qb_map[n_id][qb_num]] = int(bin_state[qb_num])

        return res

    def getSample(self, A: Operator, G: Operator,
                        evidence: dict[Union[str, int]: int],
                        optimisation_level: int = None,
                        verbose: int = 0) -> dict[int: int]:
        """
            Generates a single sample from a quantum circuit based on provided 
            evidence using amplitude amplification. (Algorithm 1)
        
            Parameters
            ----------
            A : Operator
                The quantum operator representing the initial state preparation.
            G : Operator
                The Grover iterate operator used for searching.
            evidence : dict[Union[str, int]: int]
                Dictionary with variable IDs as keys and their desired state as values.
            optimisation_level : int, optional
                Optimization level for the quantum circuit simulation, with a default setting of 1.
                Optimisation level for generate_preset_pass_manager fuction, ranges from 0 to 3
        
            Returns
            -------
            dict[int, int]
                Dictionary with variable IDs as keys and their states as values after sampling.
            """
        k = -1

        while True:
            k = k + 1

            circuit = QuantumCircuit(*list(self.q_registers.values()))
            circuit.compose(A, inplace=True)

            for i in range(2**k):
                circuit.compose(G, inplace=True)

            circuit.measure_all()

            run_res = self.qbn.aerSimulation(circuit, optimisation_level, 1)

            run_res = {self.qbn.bn.nodeId(self.qbn.bn.variable(node)): state.index(1.0)
                       for node, state in run_res.items()}

            if verbose > 0: print(f"run_res = {run_res}")

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

    def rejectionSamplingV1(self, evidence: dict[Union[str, int]: int],
                                num_samples: int = 1000,
                                verbose : int = 0) \
                                -> dict[Union[str, int]: list[float]]:
        """Performs rejection sampling on Quantum Circuit representation of
        Baysian Network

        Parameters
        ----------
        evidence: dict[Union[str, int]: int]
            Dictionary with variable IDs as keys and their state as values
        num_samples: int = 1000
            Number of samples to generate.

        Returns
        -------
        dict[Union[str, int]: list[float]]
            Dictionary with variable names as keys and proability vector as values
        """

        evidence_n_id = {self.qbn.bn.nodeId(self.qbn.bn.variable(key)): val
                         for key, val in evidence.items()}
        evidence_qbs = self.getEvidenceQuBits(evidence_n_id)

        A = self.getA()
        G = self.getG(A, evidence_qbs)

        res = {node: [0] * self.qbn.bn.variable(node).domainSize()
               for node in self.qbn.n_qb_map.keys()}

        for i in range(num_samples):
            sample = self.getSample(A, G, evidence)

            for node, state in sample.items():
                res[node][state] += 1.0/num_samples

            if verbose > 0: print(f"sample {i} \t = {sample}")
        return res

    def rejectionSampling(self, evidence: dict[Union[str, int]: int],
                                num_samples: int = 1000,
                                verbose : int = 0) \
                                -> dict[Union[str, int]: list[float]]:
        """Performs rejection sampling on Quantum Circuit representation of
        Baysian Network

        Parameters
        ----------
        evidence: dict[Union[str, int]: int]
            Dictionary with variable IDs as keys and their state as values
        num_samples: int = 1000
            Number of samples to be generated.

        Returns
        -------
        dict[Union[str, int]: list[float]]
            Dictionary with variable names as keys and proability vector as values
        """

        evidence_n_id = {self.qbn.bn.nodeId(self.qbn.bn.variable(key)): val
                         for key, val in evidence.items()}
        evidence_qbs = self.getEvidenceQuBits(evidence_n_id)

        A = QuantumCircuit(*list(self.q_registers.values()))
        self.addA(A)

        G = QuantumCircuit(*list(self.q_registers.values()))
        self.addG(G, A, evidence_qbs, inplace=False)

        res = {node: [0] * self.qbn.bn.variable(node).domainSize()
               for node in self.qbn.n_qb_map.keys()}

        for i in range(num_samples):
            sample = self.getSample(A, G, evidence)

            for node, state in sample.items():
                res[node][state] += 1.0/num_samples

            if verbose > 0: print(f"sample {i} \t = {sample}")
        return res

