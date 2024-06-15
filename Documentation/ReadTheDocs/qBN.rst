qBN package
===========

qBNMC module
------------

The `qBayesNet` class is designed to build a quantum circuit representation of a Bayesian Network. This implementation is based on the concepts presented in the paper `Quantum Circuit Representation of Bayesian Networks <https://arxiv.org/abs/2004.14803>`_ by Sima E. Borujeni. The class allows for the creation of a quantum circuit using Qiskit and facilitates performing inference on the network to compare the performance with classical Bayesian networks.

Each function in this class corresponds to specific equations and figures in the paper, providing a direct mapping of theoretical concepts to practical implementation:

- `mapNodeToQBit`: Maps variable IDs from Bayesian Network to qubit IDs for quantum circuit implementation (related to section 4.1 of the document).
- `getWidth`: Determines the number of qubits needed to represent a variable (related to section 4.1).
- `getBinarizedParameters`: Provides binary representations of variable states (related to section 4.5).
- `multiQubitRotation`: Adds rotations to the quantum circuit to map probabilities to qubits (related to section 4.4).
- `buildCircuit`: Constructs the full quantum circuit representation of the Bayesian Network (illustrated in section 4.1).

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

The `qInference` class is used to perform inference via rejection sampling from a Quantum Circuit representation of a Bayesian Network. The class leverages Qiskit to perform quantum inference, comparing its efficiency with classical methods.

Each function in this class corresponds to specific equations and concepts in the paper, providing a clear link between theory and practice : EN TRAVAUX 

.. automodule:: qBN.qBNRejection
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

