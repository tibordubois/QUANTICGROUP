import pyAgrum as gum
import numpy as np
from qiskit import QuantumRegister, AncillaRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import RYGate, XGate

class qBN:
    def __init__(self, bn):
        self.bn = bn

    def getTheta(self, id, params=[]):
        """
        Returns the rotation angle theta of a binaray variable (eq14)
        Takes pyAgrum.BayesNet object, variable id, list of parent states
        """
        if len(self.parents(id)) != len(params):
            raise NameError("params length must match number of parents")

        if self.parents(id) == set():
            P0 = self.cpt(id)[0]
            P1 = self.cpt(id)[1]

        else:
            cpt_arr = self.cpt(id).toarray()
            for val in params:
                cpt_arr = cpt_arr[val]
            P0 = cpt_arr[0]
            P1 = cpt_arr[1]

        if P0 == 0: #division by 0 check
            return np.pi
        else:
            return 2*np.arctan(np.sqrt(P1/P0))

    def getBinarizedProbability(self, binary_index, width, probability_list):
        """
        Returns the probabilities of P = 1 in a binary contex (eq18)
        Binary_index is the index of the char in the string, depends on the width (c.f. numpy.binary_repr)
        """
        all_combinations = [np.binary_repr(i, width=width) for i in range(len(probability_list))]
        target_indices = [int(bid, 2) for bid in all_combinations if bid[binary_index] == '1']
        where_specifier = [(i in target_indices) for i in range(len(probability_list))]
        return np.sum(probability_list, where=where_specifier)

    def getThetaDiscrete(self, id, params=[]):
        """
        Computes the rotation angle theta of multi state discrete variables (eq18)
        Returns a list of rotation for each of the qubits encoding the discrete variable in binary
        Takes pyAgrum.BayesNet object, variable id, list of parent states
        """
        if len(self.parents(id)) != len(params):
            raise NameError("params length must match number of parents")

        domain_size = self.variable(id).domainSize()
        #binary_rep = np.binary_repr(state, width=np.ceil(np.log2(domain_size)))

        if self.parents(id) == set():
            probability_list = [self.cpt(id)[i] for i in range(domain_size)]

        else: #no need for binarization because conditional probs are taken directly from cpt
            cpt_arr = self.cpt(id).toarray()
            for val in params:
                cpt_arr = cpt_arr[val]
            probability_list = [cpt_arr[i] for i in range(domain_size)]

        theta_list = []
        width = int(np.ceil(np.log2(domain_size)))

        for binary_index in range(width):
            P1 = getBinarizedProbability(binary_index, width, probability_list) #TO BE COMPLETED
            if P1 == 1:
                theta_list.append(np.pi)
            else:
                theta_list.append(2*np.arctan(np.sqrt(P1/(1-P1))))

        return theta_list

    def getNumQBits(self, id):
        """
        Returns the number of qubits required to represent variable id (eq21)
        """
        return np.sum([np.ceil(np.log2(bn.variable(p_id).domainSize())) for p_id in bn.parents(id)], dtype=int)

    def getTotNumQBits(self):
        """
        Returns the total number of qubits required to build the quantum circuit from bn (eq22)
        """
        s = np.sum([np.ceil(np.log2(bn.variable(id).domainSize())) for id in self.nodes()], dtype=int)
        m = np.max([(self.getNumQBits(id) + np.ceil(np.log2(self.variable(id).domainSize())) - 1) for id in self.nodes()])
        return int(s + m)

    def getRootNodes(self):
        """
        Returns the set of root nodes
        """
        return {id for id in bn.nodes() if bn.parents(id) == set()}

    def getAllParentSates(self, id):
        """
        Returns the cartesian product of the states of the parents of variable id (all the possible parent state configurations)
        """
        parent_states = [list(self.variable(pid).domain()) for pid in self.bn.parents(id)]  # Modification ici, en supposant que .domain() retourne une liste ou un ensemble directement utilisable
        return np.array(np.meshgrid(*parent_states)).T.reshape(-1, len(parent_states))


    def build_circuit(self):
        """
        Build quantum cicuit representation of a baysian networks
        Takes pyAgrum.BayesNet object
        Returns qiskit.QuantumCircuit object
        """
        quantum_register = QuantumRegister(self.getTotNumQBits())
        ancillia_register = AncillaRegister(0) # to be reconsidered
        classical_register = ClassicalRegister(1)

        circuit = QuantumCircuit(quantum_register, ancillia_register, classical_register)
        root_nodes = self.getRootNodes()
        internal_nodes = self.nodes().difference(root_nodes)

        for id in root_nodes:
            rotation = RYGate(self.getTheta(id))
            circuit.append(rotation, [id])


        for id in internal_nodes:

            parent_list = list(self.parents(id))

            print(self.parents(id))
            print(self.getAllParentSates(id))

            for params in self.getAllParentSates(id):

                circuit.barrier()

                for i in np.array(parent_list)[np.where(np.array(params)==0)]:
                    circuit.append(XGate(), [i])

                rotation = RYGate(self.getTheta(id, params)).control(len(parent_list))
                circuit.append(rotation, parent_list + [id])

                for i in np.array(parent_list)[np.where(np.array(params)==0)]:
                    circuit.append(XGate(), [i])

        circuit.barrier()

        for id in self.nodes():
            circuit.measure(id, 0)

        return circuit


