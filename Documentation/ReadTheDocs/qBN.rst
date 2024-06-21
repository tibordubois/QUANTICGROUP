qBN package
===========

qBNMC module
------------

The `qBNMC` class is designed to build a quantum circuit representation of a Bayesian Network. This implementation is based on the concepts presented in the paper `Quantum Circuit Representation of Bayesian Networks <https://arxiv.org/abs/2004.14803>`_ by Sima E. Borujeni. The class allows for the creation of a quantum circuit using Qiskit and facilitates performing inference on the network to compare the performance with classical Bayesian networks.

Each function in this class corresponds to specific equations and figures in the paper, providing a direct mapping of theoretical concepts to practical implementation:

- `mapNodeToQBit`: Maps variable IDs from Bayesian Network to qubit IDs for quantum circuit implementation (related to Section 3).
- `getWidth`: Determines the number of qubits needed to represent a variable (related to Section 3.1).
- `getBinarizedParameters`: Provides binary representations of variable states (related to Section 3.3).
- `multiQubitRotation`: Adds rotations to the quantum circuit to map probabilities to qubits (related to Section 3.2 and Figure 5).
- `buildCircuit`: Constructs the full quantum circuit representation of the Bayesian Network (illustrated in Section 3 and demonstrated with examples in Section 4).

The paper details the principles of representing a Bayesian network using quantum circuits, where nodes with discrete states are mapped to qubits, and their marginal and conditional probabilities are encoded using quantum gates such as RY and controlled-RY gates. This method allows for leveraging the advantages of quantum computing, such as superposition and entanglement, to perform inference more efficiently than classical methods.


.. automodule:: qBN.qBNMC
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

qBNRT module
------------

.. automodule:: XPs.qBNRT
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

qBNRejection module
-------------------

The `qBNRejection` class implementation is based on the principles and algorithms presented in the paper `Quantum Inference on Bayesian Networks <https://arxiv.org/abs/1402.7359>`_ by Guang Hao Low et al. The class provides a comprehensive framework for performing quantum inference, integrating key concepts from the paper into its methods. Below is a detailed explanation of how the class aligns with the paper:

- **Quantum Representation of Bayesian Networks**:
  - The class constructs the quantum operator \( A \) using the method `getA`, which builds a quantum circuit representing the Bayesian network's joint distribution, as discussed in Section VI.B of the paper.

- **Amplitude Amplification**:
  - The methods `getAdjoint`, `getB`, `getZ`, `getS`, and `getG` implement the components required for the Grover iterate, which is central to the amplitude amplification process described in Section IV.

- **Quantum Rejection Sampling Algorithm**:
  - The method `makeInference` orchestrates the rejection sampling process using amplitude amplification. This algorithm is detailed in Section V and Algorithm 1 of the paper.

- **Phase Flip Operators**:
  - The construction of phase flip operators, which are crucial for amplitude amplification, is implemented in the methods `getB`, `getZ`, and `getS`. These methods align with the explanations in Section VI.C.

- **Evidence Handling and Sampling**:
  - The method `getEvidenceQuBits` maps classical evidence to quantum states, preparing them for quantum operations. The `getSample` method utilizes amplitude amplification to generate samples that match the evidence, following the procedure outlined in Section V.


.. automodule:: qBN.qBNRejection
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

