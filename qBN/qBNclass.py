import pyAgrum as gum

import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import RYGate, XGate

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import StatevectorSampler

from typing import Union,List,Dict

class qBayesNet:

    def __init__(self, bn:"pyAgrum.BayesNet"):
        self.bn = bn

    def getWidth(self, node: Union[str, int]) -> int:
        """
        (eq21)

        Parameters
        ---------
        node: Union[str, int]
            Name or id of the corresponding node

        Returns
        -------
        int
            Number of qubits required to represent
        """

        domain_size = self.bn.variable(node).domainSize()
        return int(np.ceil(np.log2(domain_size)))

    def getTotNumQBits(self) -> int:
        """
        (eq21)

        Parameters
        ---------

        Returns
        -------
        int
            Total number of qubits required to build the quantum circuit
        """
        s = np.sum([self.getWidth(id) for id in self.bn.nodes()], dtype=int)
        return int(s)

    def getRootNodes(self) -> set[int]:
        """
        Parameters
        ---------

        Returns
        -------
        set
            Set of int representing to root nodes id
        """
        return {id for id in self.bn.nodes() if len(self.bn.parents(id)) == 0}

    def getAllParentSates(self, node: Union[str, int]) -> list[dict]: #should return type [{"A":0, "B":1}, ... ] #UNIQUELY
        """
        Returns all the possible parent states of node

        Parameters
        ---------
        node: Union[str, int]
            Name or id of the corresponding node

        Returns
        -------
        List[Dict]
            List containting all the CPT with the node column dropped represented in a dictionnary
        """
        res = list()

        I=gum.Instantiation()
        for n in self.bn.cpt(node).names[1:]:
            I.add(self.bn.variable(n))

        I.setFirst()
        while not I.end():
            res.append(I.todict())
            I.inc()

        return res

    def mapNodeToQBit(self) -> Dict[int, List[int]]:
        """
        Maps node from baysian network to a number of qubits ids depending on the node domain size

        Parameters
        ---------

        Returns
        -------
        dict[int: list[int]]
            Dictionary with the node id as key and a list of corresponding qubit ids as value
        """
        res = dict()
        qubit_id = 0
        for n_id in self.bn.nodes():
            res[n_id] = []
            for state in range(self.getWidth(n_id)):
                res[n_id].append(qubit_id)
                qubit_id = qubit_id + 1

        return res

    def getBinarizedProbability(self, binary_index: int, width: int, probability_list: list[float]) -> float:
        """
        (eq18)

        Parameters
        ---------
        binary_index: int
            Index of the char in the string
        width: int
            (c.f. numpy.binary_repr)
        probability_list: list[float]
            list of the probabilities in BN

        Returns
        -------
        float
            Returns the probabilities of P = 1 in a binary context
        """
        all_combinations = [np.binary_repr(i, width=width) for i in range(len(probability_list))]
        target_indices = [int(bid, 2) for bid in all_combinations if bid[binary_index] == '1']
        where_specifier = [(i in target_indices) for i in range(len(probability_list))]
        return np.sum(probability_list, where=where_specifier)

    def getTheta(self, node: Union[str, int], params: dict[int: int] = None) -> list[float]:
        """
        (eq18)

        Parameters
        ---------
        node: Union[str, int]
            Name or id of the corresponding node
        params: dict[int: int] = None
            states of parent states

        Returns
        -------
        list[float]
            List of roation for each of the qubits encoding the discrete variable in binary
        """
        if params is None:
            params=dict()

        if len(self.bn.parents(node)) != len(params):
            raise NameError("params length must match number of parents")

        probability_list = self.bn.cpt(node)[params].tolist()

        theta_list = []
        width = self.getWidth(node)

        for binary_index in range(width):
            P1 = self.getBinarizedProbability(binary_index, width, probability_list)
            theta_list.append(np.pi) if P1 == 1 else theta_list.append(2*np.arctan(np.sqrt(P1/(1-P1))))

        return theta_list

    def getBinarizedParameters(self, width_dict: dict[Union[str,int]:int], param_dict: dict[Union[str,int]:int]) -> dict[str:list[int]]:
        """
        Parameters
        ---------
        width_list: dict[Union[str,int]:int]
            Dict where keys are nodes and the values their corresponding width (c.f. getWidth())
        param_dict: dict[Union[str,int]:int]
            Dict where keys are nodes and the values their corresponding states


        Returns
        -------
        list[int]
            List containing the associated binarized paramters (0, 1) indexed regularly
        """
        width_dict_id = {self.bn.nodeId(self.bn.variable(key)):val for key, val in width_dict.items()}
        param_dict_id = {self.bn.nodeId(self.bn.variable(key)):val for key, val in param_dict.items()}
        bin_params_dict = dict()
        for id in param_dict_id.keys():
            bin_params_dict[id] =  np.array(list(np.binary_repr(param_dict_id[id], width=width_dict_id[id]))).astype(int)
        return bin_params_dict

    def getQuantumRegisters(self) -> dict[int: QuantumRegister]:
        """
        Parameters
        ---------

        Returns
        -------
        dict[int: QuantumRegister]
            Dictionnary with node id as keys and quantum registers as values
        """
        return {n_id: QuantumRegister(int(np.ceil(np.log2(self.bn.variable(n_id).domainSize()))), n_id) for n_id in self.bn.nodes()}

    def buildCircuit(self) -> QuantumCircuit:
        """
        

        Returns
        -------
        QuantumCircuit
        """

        q_reg_dict = self.getQuantumRegisters()

        circuit = QuantumCircuit(*list(q_reg_dict.values()))

        root_nodes = self.getRootNodes()
        internal_nodes = self.bn.nodes().difference(root_nodes)

        n_qb_map = self.mapNodeToQBit()

        for n_id in root_nodes:
            theta_list = self.getTheta(n_id)
            for qb_number in range(len(n_qb_map[n_id])): #qubit number is NOT qubit id, for implementational constaint reasons with list getThetaDiscrete
                rotation = RYGate(theta_list[qb_number])
                circuit.append(rotation, [n_qb_map[n_id][qb_number]])

        for n_id in internal_nodes:

            parent_id_set = self.bn.parents(n_id)
            parent_qbit_list = np.ravel([n_qb_map[p_id] for p_id in parent_id_set]) #list containing qubit id of each of the parents in order

            for params_dict in self.getAllParentSates(n_id): #params is dict

                width_dict = {p_id: self.getWidth(p_id) for p_id in parent_id_set}

                bin_params = self.getBinarizedParameters(width_dict, params_dict)

                theta_list = self.getTheta(n_id, params_dict)

                circuit.barrier()

                for ctrl_qb_id in np.array(parent_qbit_list)[np.where(np.ravel(list(bin_params.values()))==0)]:
                    circuit.append(XGate(), qargs=[ctrl_qb_id])

                for qb_number in range(len(n_qb_map[n_id])):
                    rotation = RYGate(theta_list[qb_number]).control(len(parent_qbit_list))
                    circuit.append(rotation, qargs=list(parent_qbit_list)+[n_qb_map[n_id][qb_number]])

                for ctrl_qb_id in np.array(parent_qbit_list)[np.where(np.ravel(list(bin_params.values()))==0)]:
                    circuit.append(XGate(), qargs=[ctrl_qb_id])

        circuit.measure_all()

        return circuit

    def run(self, shots: int = 1024) -> dict[Union[str, int]: dict[int: float]]:
        """
        Parameters
        ---------
        shots: int = 1024
            Number of times to be run

        Returns
        -------
        dict[Union[str, int]: dict[int: float]]
            Dictionnary with keys as Variable names, and values as CPT as Dict
        """
        qbn = self.buildCircuit()
        sampler = StatevectorSampler()

        pm = generate_preset_pass_manager(optimization_level=1)
        isa_circuit = pm.run(qbn)

        result = sampler.run([isa_circuit], shots=shots).result()

        data_pub = result[0].data

        bitstrings = data_pub.meas.get_bitstrings()
        bitstrings = np.array([np.array(list(string[::-1])) for string in bitstrings])

        res = dict()

        n_qb_map = self.mapNodeToQBit()

        for n_id in self.bn.nodes():

            res[self.bn.variable(n_id).name()] = dict()
            width = len(n_qb_map[n_id])

            for state in range(len(self.bn.cpt(n_id).tolist())):

                pattern = list(np.binary_repr(state, width=width))

                matches = np.all(bitstrings[:, n_qb_map[n_id][0]: n_qb_map[n_id][-1]+1] == pattern, axis=1)

                res[self.bn.variable(n_id).name()][state] = np.sum(matches)/shots

        return res
