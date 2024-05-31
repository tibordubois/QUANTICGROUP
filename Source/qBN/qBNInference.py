#from old.qBNclass_lessold import qBayesNet
from qBN.qBNclass import qBayesNet
import numpy as np

from typing import Union #List and Dict are deprecated (python 3.9)

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import XGate, ZGate
from qiskit.quantum_info import Operator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import StatevectorSampler

from qiskit.visualization import array_to_latex


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

    Methods
    -------
    getA(self) -> Operator:
        Gives the Operator object represenation of the Quantum Circuit representing
        the Baysian Network
    """

    def __init__(self, qbn: qBayesNet) -> None:
        """
        Parameters
        ----------
        qbn : qBayesNet
            Quantum Bayesian Network
        """

        self.qbn = qbn
        self.q_registers = self.qbn.getQuantumRegisters()
        self.all_qbits = np.ravel(list(self.qbn.n_qb_map.values())).tolist()

    def getA(self) -> Operator:
        """Gives the quantum sample preparation Operator object
        Operator of the Quantum Circuit representing the Baysian Network

        Returns
        -------
        Operator
            Quantum gate A
        """

        circuit = self.qbn.buildCircuit(add_measure=False)
        circuit = circuit.decompose()

        A = Operator(circuit)
        A = A.to_instruction()
        A.label = 'A'
        return A

    def getAdjoint(self, M: Operator):
        """Gives the adjoint operator of M

        Parameters
        ----------
        M: Operator
            Operator to be transformed

        Returns
        -------
        Operator
            M adjoint
        """

        M_label = M.label
        M = Operator(M.adjoint())
        M = M.to_instruction()
        M.label = M_label+'\u2020'
        return M

    def getB(self, evidence_qbs: dict[int, int]) -> Operator:
        """Gives the B gate of the phase flip operator (eq7)

        Parameters
        ----------
        evidence_qbs: dict[int, int]
            Dictionary with qubit IDs as keys and their quantum state as values

        Returns
        -------
        Operator
            Quantum gate B
        """

        circuit = QuantumCircuit(*list(self.q_registers.values()))

        for qb_id, qb_state in evidence_qbs.items():
            if qb_state == 0:
                circuit.append(XGate(), [qb_id])

        circuit = circuit.decompose()

        B = Operator(circuit)
        B = B.to_instruction()
        B.label = 'B'
        return B

    def getZ(self, evidence_qbs: dict[int, int]) -> Operator:
        """Gives the Z gate of the phase flip operator (eq7)

        Parameters
        ----------
        evidence_qbs: dict[int, int]
            Dictionary with qubit IDs as keys and their quantum state as values

        Returns
        -------
        Operator
            Quantum gate Z
        """

        circuit = QuantumCircuit(*list(self.q_registers.values()))

        rotation = ZGate()
        if len(evidence_qbs) > 1:
            rotation = rotation.control(len(evidence_qbs) - 1)

        circuit.append(rotation, list(evidence_qbs.keys()))

        circuit = circuit.decompose()

        Z = Operator(circuit)
        Z = Z.to_instruction()
        Z.label = 'Z'
        return Z

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

        B = self.getB(evidence_qbs)
        Z = self.getZ(evidence_qbs)

        circuit.append(B, qargs=self.all_qbits)
        circuit.append(Z, qargs=self.all_qbits)
        circuit.append(B, qargs=self.all_qbits)

        evidence_string = ''.join([str(q_state) for q_state in evidence_qbs.values()])

        circuit = circuit.decompose()

        S = Operator(circuit)
        S = S.to_instruction()
        label = 'S'+evidence_string

        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        label = label.translate(SUB)

        S.label = label

        return S

    def getG(self, A: Operator, evidence_qbs: dict[int, int]) -> Operator:
        """Gives the grover iterate

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

        circuit = QuantumCircuit(*list(self.q_registers.values()))

        Se = self.getS(evidence_qbs)
        S0 = self.getS({qb_id: 0 for qb_id in self.all_qbits})

        A_adj = self.getAdjoint(A)

        circuit.append(Se, qargs=self.all_qbits)
        circuit.append(A_adj, qargs=self.all_qbits)
        circuit.append(S0, qargs=self.all_qbits)
        circuit.append(A, qargs=self.all_qbits)

        circuit = circuit.decompose()

        G = Operator(circuit)
        G = G.to_instruction()
        G.label = 'G'

        return G

    def getEvidenceQuBits(self, evidence: dict[int: int]) -> dict[int, int]:
        """Gives qubit representation of evidence in Baysian Network

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
                        optimisation_level: int = None,
                        verbose: int = 0) -> dict[int: int]:
        """Generate one sample from evidence (Algorithm 1)

        Parameters
        ----------
        A: Operator
            Gate A
        G: Operator
            Gate G
        evidence: dict[Union[str, int]: int]
            Dictionary with variable IDs as keys and their state as values
        optimisation_level: int = 1
            Optimisation level for generate_preset_pass_manager fuction,
            ranges from 0 to 3

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

            for i in range(int(np.ceil(2.0**k))):
                circuit.compose(G, inplace=True)

            circuit.measure_all(add_bits=False)

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
            Number of samples

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
