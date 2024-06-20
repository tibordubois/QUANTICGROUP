# qBN README

## Overview

The `qBN` package is designed for representing and performing inference on Bayesian Networks using Quantum Circuits. It leverages the principles of quantum information theory to implement quantum versions of classical probabilistic reasoning algorithms, enabling potentially significant speedups for certain types of computations.

### Structure

The package consists of two main classes:

1. **qBNRejection**: This class performs inference via rejection sampling from a Quantum Circuit representation of a Bayesian Network.
2. **qBNMC**: This class constructs a Quantum Circuit representation of a Bayesian Network.

## Implementation

### qBNMC

The `qBNMC` class is responsible for building the quantum circuit that represents the Bayesian Network. This involves mapping the variables and their states in the Bayesian Network to qubits and their corresponding quantum states. The quantum circuit is constructed in a way that reflects the probabilistic relationships encoded in the Bayesian Network.

### qBNRejection

The `qBNRejection` class utilizes the quantum circuit constructed by `qBNMC` to perform inference. This is done using a process called rejection sampling, where samples are generated from the quantum circuit and then used to compute the probabilities of different states in the Bayesian Network given some evidence.

### Research Basis

The implementation of this package is based on the following research papers:

1. **Quantum Inference on Bayesian Networks** by Guang Hao Low: This paper discusses the theory and methods for performing quantum inference on Bayesian Networks.
2. **Quantum circuit representation of Bayesian networks** by Sima E. Borujeni: This paper outlines the methodology for constructing quantum circuits that represent Bayesian Networks.

## Usage

To use the `qBN` package, follow these steps:

1. **Install the required dependencies**:

   ```bash
   pip install numpy qiskit scipy pyAgrum qiskit-aer qiskit-ibm-runtime
   ```

2. **Create and initialize a Bayesian Network**:

   ```python
   from pyAgrum import BayesNet
   from qBN.qBNMC import qBNMC, qBNRejection

   # Create a Bayesian Network
   bn = BayesNet()

   # Add nodes and edges to the Bayesian Network
   bn.add('A', 2)
   bn.add('B', 2)
   bn.addArc('A', 'B')

   # Create a Quantum Bayesian Network
   qbn = qBNMC(bn)
   ```

3. **Perform inference using the quantum circuit**:

   ```python
   # Create a Quantum Inference object
   qBNRejection = qBNRejection(qbn)

   # Set evidence for the inference
   qBNRejection.setEvidence({'A': 1})

   # Perform inference
   results = qBNRejection.makeInference()

   # Print the results
   print(results)
   ```

## Testing

In the main directory, the folder `tutorials` has a Jupyter-Notebook that illustrates an exhaustive example of using all the functions from the package on the ASIA Bayesian Network, which is a standard test network provided by the pyAgrum library.

### Running the Tests

To run the tests, you can execute the `test_qBN.py` script. This will demonstrate the full functionality of the `qBN` package, including each and every methods that is contained in qBNMC and qBNRejection.

## Conclusion

The `qBN` package provides a powerful tool for leveraging quantum computing in probabilistic reasoning tasks. By integrating Bayesian Networks with quantum circuits, it opens up new possibilities for faster and more efficient inference, especially for large and complex networks. For detailed information on the implementation and theoretical background, please refer to the research papers references below.

## Authors

- [Dubois Tibor, Rioual Thierry, Gunes Mehmet]

## References

- Sima E. Borujeni, ["Quantum circuit representation of Bayesian networks"](https://arxiv.org/pdf/2004.14803)
- Guang Hao Low, Theodore J. Yoder, Isaac L. Chuang, ["Quantum Inference on Bayesian Networks"](https://arxiv.org/pdf/1402.7359)
- Documentation de [pyAgrum](https://pyagrum.org/)
- Documentation de [Qiskit](https://qiskit.org/)
