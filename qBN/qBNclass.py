from pyAgrum import BayesNet, Instantiation, Potential

import numpy as np

from typing import Union #List and Dict are deprecated (python 3.9)

from qiskit import QuantumRegister, QuantumCircuit, transpile
from qiskit.circuit.library import RYGate, XGate
from qiskit.quantum_info import Operator
from qiskit.visualization import plot_distribution

from qiskit_aer import AerSimulator

from qiskit_ibm_runtime import SamplerV2

class qBayesNet:
    """
    Class used to build a Quantum Circuit representation of a Bayesian Network
    Based on the paper:
    Quantum circuit representation of Bayesian networks - Sima E. Borujeni

    Attributes
    ----------
    bn: BayesNet
        pyAgrum Bayesian Network
    n_qb_map: dict[int: list[int]]
        Dictionary where keys are IDs of variables in bn and values are
        the list of IDs of the associated qubits

    Methods
    -------
    mapNodeToQBit(self) -> dict[int: list[int]]
        Maps variables IDs from Baysian Network to a list of qubits IDs
        to be implemented in a Quantum Circuit

    getWidth(self, node: Union[str, int]) -> int:
        Give the required number of qubits to represent a variable

    getBinarizedParameters(self, width_dict: dict[Union[str, int]: int],
                                 param_dict: dict[Union[str, int]: int])
                                 -> dict[Union[str, int]: list[int]]
        Gives the binary representations of the states of a variable
        in a Bayesian Network

    getRootNodes(self) -> set[int]:
        Gives the IDs of the root nodes (variables) in the DAG representation
        of the Bayesian Network

    getAllParentSates(self, node: Union[str, int]) \
                            -> list[dict[Union[str, int]: int]]:
        Gives all the possible parent state combinations of a given variable

    getQuantumRegisters(self) -> dict[int: QuantumRegister]:
        Gives the Quantum Registers used to represent the Bayesian Network
        as a Quantum Circuit

    indicatorFunction(self, binary_list: list[list[int]],
                            targets: dict[int, int], verbose: int = 0)
                            -> list[bool]:
        Gives a list of matches when given a list of binary strings
        (represented using lists) and a dictionary of conditions
        (eq17) (eq19)

    getProbability(self, value: int,
                         node: Union[str, int],
                         qb_id: int,
                         params_qb: dict[int, int],
                         params_node: dict[Union[str, int]: int] = None,
                         verbose: int = 0) -> float:
        Gives the probability that the qubit with given ID (in the context of
        the whole circuit) equals to the given value conditioned to other
        qubits representing the variable and other nodes in the Bayesian Network
        (eq18) (eq20)

    multiQubitRotation(self, circuit: QuantumCircuit,
                             node: Union[str, int],
                             target_qb: list[int],
                             params_qb: dict[int, int],
                             params_node: dict[Union[str, int]: int] = None,
                             control_qb: list[int] = None,
                             verbose: int = 0) -> Operator:
        Procedure that adds to the Quantum Circuit a series of rotations
        that maps the probabilities of the variable to the qubits representing it
        (Fig9) (eq18)

    buildCircuit(self, verbose: int = 0) -> QuantumCircuit:
        Builds the Quantum Circuit representation of Bayesian Network

    run(self, shots: int = 8192) -> dict[Union[str, int]: dict[int: float]]:
        Builds and runs the quantum circuit representation of a bayesian network

    """

    def __init__(self, bn: BayesNet, target_nodes: set[Union[str, int]] = None) -> None:
        """
        Parameters
        ----------
        bn : BayesNet
            pyAgrum Bayesian Network
        """

        self.target_nodes = bn.nodes() if target_nodes == None else target_nodes
        self.target_nodes = {bn.nodeId(bn.variable(name)) for name in self.target_nodes}

        self.bn = bn
        self.n_qb_map = self.mapNodeToQBit(self.target_nodes)

    def mapNodeToQBit(self, nodes: set[int]) -> dict[int: list[int]]:
        """Maps variables IDs from Baysian Network to a list of qubits IDs
        to be implemented in a Quantum Circuit

        Returns
        -------
        dict[int: list[int]]
            Dictionary with variable ID as key and the list of its corresponding
            qubit IDs as value
        """

        res = dict()
        qubit_id = 0
        for n_id in nodes:
            res[n_id] = []
            for i in range(self.getWidth(n_id)):
                res[n_id].append(qubit_id)
                qubit_id = qubit_id + 1

        return res

    def getWidth(self, node: Union[str, int]) -> int:
        """Give the required number of qubits to represent a variable
        (eq21)

        Parameters
        ----------
        node: Union[str, int]
            Name or id of the corresponding node

        Returns
        -------
        int
            Number of qubits required to represent the given variable
        """

        domain_size = self.bn.variable(node).domainSize()
        return int(np.ceil(np.log2(domain_size)))

    def getTotNumQBits(self) -> int: #not used
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

    def getBinarizedParameters(self, width_dict: dict[Union[str, int]: int],
                                     param_dict: dict[Union[str, int]: int]) \
                                     -> dict[Union[str, int]: list[int]]:
        """Gives the binary representations of the states of a variable in a
        Bayesian Network

        Parameters
        ----------
        width_list: dict[Union[str,int]:int]
            Dictionary with name of variables as keys and their corresponding widths
            as values (c.f. getWidth())
        param_dict: dict[Union[str,int]:int]
            Dictionary with name of variables as keys and their states as values

        Returns
        -------
        dict[str: list[int]]
            Dictionary with name of variables as keys and a binary string
            (list of 0s and 1s) as values
        """

        width_dict_id = {self.bn.nodeId(self.bn.variable(key)): val
                         for key, val in width_dict.items()}
        param_dict_id = {self.bn.nodeId(self.bn.variable(key)): val
                         for key, val in param_dict.items()}
        bin_params_dict = dict()
        for id in param_dict_id.keys():
            bin_params_dict[id] = np.array(
                list(np.binary_repr(param_dict_id[id], width=width_dict_id[id]))
            ).astype(int).tolist()
        return bin_params_dict

    def getRootNodes(self) -> set[int]:
        """Gives the IDs of the root nodes (variables) in the DAG representation of
        the Bayesian Network

        Returns
        -------
        set[int]
            Set of integers representing root node IDs
        """

        return {id for id in self.bn.nodes() if len(self.bn.parents(id)) == 0}

    def getAllParentSates(self, node: Union[str, int]) \
                                -> list[dict[Union[str, int]: int]]:
        """Gives all the possible parent state combinations of a given variable

        Parameters
        ----------
        node: Union[str, int]
            Name or id of the corresponding node

        Returns
        -------
        list[dict[Union[str, int]: int]]
            List containting dicrionnaries with variable names as keys and their
            corresponding state as values
        """

        res = list()

        inst = Instantiation()
        for name in self.bn.cpt(node).names[1:]:
            n_id = self.bn.nodeId(self.bn.variable(name))
            if n_id in self.target_nodes:
                inst.add(self.bn.variable(name))

        inst.setFirst()
        while not inst.end():
            res.append(inst.todict())
            inst.inc()

        return res

    def getQuantumRegisters(self) -> dict[int: QuantumRegister]:
        """Gives the Quantum Registers used to represent the Bayesian Network as
        a Quantum Circuit

        Returns
        -------
        dict[int: QuantumRegister]
            Dictionary with variable IDs as keys and Quantum Registers as values
        """

        res = dict()

        for  n_id in self.target_nodes:
            res[n_id] = QuantumRegister(
                int(np.ceil(np.log2(self.bn.variable(n_id).domainSize()))), n_id)
        return res

    def indicatorFunction(self, binary_list: list[list[int]],
                                targets: dict[int, int],
                                verbose: int = 0) -> list[bool]:
        """Gives a list of matches when given a list of binary strings
        (represented using lists) and a dictionary of conditions
        (eq17) (eq19)

        Parameters
        ----------
        binary_list: list[list[int]]
            List of lists of 0s and 1s representing the basis states that forms
            the superposition state representing a variable in the context of 
            (eq16)
        targets: dict[int, int]
            Dictionary with the indices of qubits (relative to the binary list and
            not the whole quantum circuit) as keys and their corresponding
            binary states as values

        Returns
        -------
        list[bool]
            List where True corresponds to a binary string (list of 0s and 1s)
            satisfying the condition given by targets, and False otherwise
        """

        if verbose > 0:
            print(f"\nindicatorFunction call: \
                  binary_list = {binary_list}, \
                  targets = {targets}")

        if len(targets) == 0:
            return [False] * len(binary_list)

        sorted_target_keys = sorted(list(targets))
        binary_arr = np.array(binary_list)

        if verbose > 0:
            print(f"STK = {sorted_target_keys}")
            print(f"binary_arr = \n{binary_arr}")

        binary_arr = binary_arr[:,sorted_target_keys]
        pattern = [targets[key] for key in sorted_target_keys]
        matches = np.all(binary_arr == pattern, axis=1)

        return list(matches)

    def getProbability(self, value: int, 
                             node: Union[str, int], 
                             qb_id: int, 
                             param_qbs: dict[int, int], 
                             param_nodes: dict[Union[str, int]: int] = None, 
                             verbose: int = 0) -> float:
        """Gives the probability that the qubit with given ID (in the context of
        the whole circuit) equals to the given value conditioned to other qubits
        representing the variable and other nodes in the Bayesian Network
        (eq18) (eq20)

        Parameters
        ----------
        value: int
            0 or 1
        node: Union[str, int]
            Name or id of the corresponding variable
        qb_id: int
            Global index of the qubit in the Quantum Circuit
        param_qbs: dict[int, int]
            Dictionary with global qubit index as keys and quantum state as values,
            the qubits are representing the same variable as the main qubit
        param_nodes: dict[Union[str, int]: int] = None
            Dictionary with variable name as keys and their state as values

        Returns
        -------
        float
            Value of the probability measure
        """

        if verbose > 0:
            print(f"\ngetP1 call : \
                  node = {node}, \
                  qb_id = {qb_id}, \
                  param_nodes = {param_nodes}")

        if param_nodes is None:
            param_nodes = dict()

        probability_list = self.bn.cpt(node)[param_nodes] #of BN
        width = self.getWidth(node)
        number_of_states = self.bn.cpt(node).shape[0]
        binary_state_list = [np.array(
            list(np.binary_repr(state, width=width)), dtype=int).tolist()
            for state in range(number_of_states)] #list of all states in binary

        qb_number = self.n_qb_map[node].index(qb_id) #relative qubit number
        params_qb_number_dict = {self.n_qb_map[node].index(key): value 
                                 for key, value in param_qbs.items()}

        if verbose > 0:
            print(f"probability_list = {probability_list}")
            print(f"binary_state_list = {binary_state_list}")
            print(f"qb_id = {qb_id}, qb_number = {qb_number}")
            print(f"param_qbs = {param_qbs}, \
                  params_qb_number_dict = {params_qb_number_dict}")


        I_qb = self.indicatorFunction(binary_state_list, 
                                      {**{qb_number: value}, 
                                       **params_qb_number_dict}, 
                                      verbose=verbose)

        if verbose > 0:
            print(f"indicator = {I_qb}")

        return np.sum(probability_list, where=I_qb)

    def multiQubitRotation(self, circuit: QuantumCircuit, 
                                 node: Union[str, int], 
                                 target_qbs: list[int], 
                                 param_qbs: dict[int, int], 
                                 param_nodes: dict[Union[str, int]: int] = None, 
                                 control_qbs: list[int] = None, 
                                 verbose: int = 0) -> Operator:
        """Procedure that adds to the Quantum Circuit a series of rotations that 
        maps the probabilities of the variable to the qubits representing it
        (Fig9) (eq18)

        Parameters
        ---------
        circuit: QuantumCircuit
            Quantum Circuit to which the rotations are added
        node: Union[str, int]
            Name or id of the corresponding node
        target_qbs: list[int]
            List containing the IDs of the qubits representing the variable in 
            the circuit
        param_qbs: dict[int, int]
            Dictionary with qubit id as keys and their quantum state as values 
            (used for recursion)
        param_nodes: dict[Union[str, int]: int]
            Dictionary with variable name as keys and their state as values
        control_qbs:
            List containing the IDs of qubits representing parent nodes of 
            the variable in DAG
        """

        if verbose > 0 :
            print(f"\nmultiQubitRotation call : node = {node}, \
                  target_qb = {target_qbs}, param_qbs = {param_qbs}, \
                  param_nodes = {param_nodes}, control_qbs = {control_qbs}")

        if param_nodes == None:
            param_nodes = dict()
        if control_qbs == None:
            control_qbs = list()

        target_copy = target_qbs.copy()

        params_qb_list = sorted(list(param_qbs))

        P1 = self.getProbability(1, node, target_copy[0], 
                                 param_qbs, param_nodes, verbose=verbose)
        P0 = self.getProbability(0, node, target_copy[0], 
                                 param_qbs, param_nodes, verbose=verbose)

        theta = np.pi if P0 == 0 else 2*np.arctan(np.sqrt(P1/P0))

        if verbose > 0:
            print(f"P1 = {P1}, P0 = {P0}")
            print(f"theta = {theta}")

        RY = RYGate(theta)
        X = XGate()


        if len(target_copy) == 1: #base case

            if len(param_qbs)+len(control_qbs) > 0:
                RY = RY.control(len(param_qbs)+len(control_qbs))

            qargs = control_qbs + params_qb_list + [target_copy[0]]
            circuit.append(RY, qargs = qargs)

        else: #recursion

            if len(param_qbs)+len(control_qbs) > 0:
                RY = RY.control(len(param_qbs)+len(control_qbs))
                X = X.control(len(param_qbs)+len(control_qbs))

            qargs = control_qbs + params_qb_list + [target_copy[0]]
            circuit.append(RY, qargs = qargs)

            popped_qb = target_copy.pop(0)

            self.multiQubitRotation(circuit, node, target_copy, 
                                    {**param_qbs, **{popped_qb: 1}}, 
                                    param_nodes, control_qbs, verbose=verbose)

            qargs = control_qbs + params_qb_list + [popped_qb]
            circuit.append(X, qargs = qargs)

            self.multiQubitRotation(circuit, node, target_copy, 
                                    {**param_qbs, **{popped_qb: 0}}, 
                                    param_nodes, control_qbs, verbose=verbose)

            qargs = control_qbs + params_qb_list + [popped_qb]
            circuit.append(X, qargs = qargs)

        if verbose > 0:
            print(circuit.draw())

        return

    def buildCircuit(self, add_measure: bool = True,
                           verbose: int = 0) -> QuantumCircuit:
        """Builds the Quantum Circuit representation of Bayesian Network

        Parameters
        ---------
        add_measure: bool = True
            Adds measurement gate for every qubit at the end of the circuit

        Returns
        -------
        QuantumCircuit
        """

        if verbose > 0:
            print("call buildCircuit")

        q_reg_dict = self.getQuantumRegisters()
        circuit = QuantumCircuit(*list(q_reg_dict.values()))

        root_nodes = self.getRootNodes()
        internal_nodes = self.bn.nodes().difference(root_nodes)

        for n_id in root_nodes.intersection(self.target_nodes):

            self.multiQubitRotation(circuit, n_id, self.n_qb_map[n_id], {}, 
                                    verbose=verbose)

        for n_id in internal_nodes.intersection(self.target_nodes):

            parent_id_set = self.bn.parents(n_id).intersection(self.target_nodes)
            parent_qbit_list = list(np.ravel([self.n_qb_map[p_id] 
                                              for p_id in parent_id_set])) 
            #list containing qubit id of each of the parents in order

            for params_dict in self.getAllParentSates(n_id): #params is dict

                width_dict = {p_id: self.getWidth(p_id) for p_id in parent_id_set}
                bin_params = self.getBinarizedParameters(width_dict, params_dict)

                circuit.barrier()

                for ctrl_qb_id in np.array(parent_qbit_list)[
                    np.where(np.ravel(list(bin_params.values()))==0)
                    ]:
                    circuit.append(XGate(), qargs=[ctrl_qb_id])

                self.multiQubitRotation(circuit, n_id, self.n_qb_map[n_id], {}, 
                                        params_dict, parent_qbit_list, 
                                        verbose=verbose)

                for ctrl_qb_id in np.array(parent_qbit_list)[
                    np.where(np.ravel(list(bin_params.values()))==0)
                    ]:
                    circuit.append(XGate(), qargs=[ctrl_qb_id])

        if add_measure:
            circuit.measure_all()

        return circuit

    def aerSimulation(self, circuit: QuantumCircuit,
                          optimization_level: int = None,
                          shots: int = 10000) -> dict[Union[str, int]: list[float]]:
        """Builds and runs quantum circuit from parameter
        Parameters
        ---------
        circuit: QuantumCircuit
            Qunatum Circuit to be run
        optimisation_level: int = None
            Optimisation level for transpile fuction ranges from 0 to 3
        shots: int = 10000
            Number of times to be run

        Returns
        -------
        dict[Union[str, int]: Potential]
            Dictionary with variable names as keys, and corresponding their
            probability vectors as values
        """

        backend_aer = AerSimulator()
        sampler_aer = SamplerV2(backend=backend_aer)
        circuit_aer = transpile(circuit, backend=backend_aer,
                                optimization_level=optimization_level)
        job_aer = sampler_aer.run([(circuit_aer, None, shots)])
        result_aer = job_aer.result()
        counts_aer = result_aer[0].data.meas.get_counts()

        res = dict()
        for n_id in self.target_nodes:
            width = len(self.n_qb_map[n_id])
            probability_vector = list()

            for state in range(self.bn.variable(n_id).domainSize()):
                pattern = np.binary_repr(state, width=width)
                matches = [val for key, val in counts_aer.items()
                           if key[::-1][self.n_qb_map[n_id][0]: self.n_qb_map[n_id][-1] + 1]
                           == pattern]
                probability_vector.append(np.sum(matches)/shots)

            res[self.bn.variable(n_id).name()] = probability_vector

        return res

    def runBN(self, optimisation_level: int = None,
                    shots: int = 10000) -> dict[Union[str, int]: Potential]:
        """Builds and runs the quantum circuit representation of a bayesian network
        Parameters
        ---------
        optimisation_level: int = 1
            Optimisation level for generate_preset_pass_manager fuction, 
            ranges from 0 to 3
        shots: int = 10000
            Number of times to be run

        Returns
        -------
        dict[Union[str, int]: Potential]
            Dictionary with variable names as keys, and corresponding Potentail as 
            values
        """

        circuit = self.buildCircuit()
        run_res = self.aerSimulation(circuit, optimisation_level, shots)
        res = dict()

        for n_id, p_vect in run_res.items():
            portential = Potential().add(self.bn.variable(n_id))
            portential.fillWith(p_vect)
            res[self.bn.variable(n_id).name()] = portential

        return res
    
    def runBNv2(self, operator: Operator, optimisation_level: int = None,
                    shots: int = 10000) -> dict[Union[str, int]: Potential]:
        """Builds and runs the quantum circuit representation of a bayesian network
        Parameters
        ---------
        optimisation_level: int = 1
            Optimisation level for generate_preset_pass_manager fuction, 
            ranges from 0 to 3
        shots: int = 10000
            Number of times to be run

        Returns
        -------
        dict[Union[str, int]: Potential]
            Dictionary with variable names as keys, and corresponding Potentail as 
            values
        """

        q_reg_dict = self.getQuantumRegisters()
        all_qubits = list(np.ravel([self.n_qb_map[p_id] for p_id in self.bn.nodes()])) 
        circuit = QuantumCircuit(*list(q_reg_dict.values()))
        circuit.append(operator, all_qubits)

        circuit.measure_all()

        run_res = self.aerSimulation(circuit, optimisation_level, shots)
        res = dict()

        for n_id, p_vect in run_res.items():
            portential = Potential().add(self.bn.variable(n_id))
            portential.fillWith(p_vect)
            res[self.bn.variable(n_id).name()] = portential

        return res
    
    def runBNv3(self, circuit: QuantumCircuit, optimisation_level: int = None,
                    shots: int = 10000) -> dict[Union[str, int]: Potential]:
        """Builds and runs the quantum circuit representation of a bayesian network
        Parameters
        ---------
        optimisation_level: int = 1
            Optimisation level for generate_preset_pass_manager fuction, 
            ranges from 0 to 3
        shots: int = 10000
            Number of times to be run

        Returns
        -------
        dict[Union[str, int]: Potential]
            Dictionary with variable names as keys, and corresponding Potentail as 
            values
        """

        run_res = self.aerSimulation(circuit, optimisation_level, shots)
        res = dict()

        for n_id, p_vect in run_res.items():
            portential = Potential().add(self.bn.variable(n_id))
            portential.fillWith(p_vect)
            res[self.bn.variable(n_id).name()] = portential

        return res

    def getRMSPE(self):
        pass
