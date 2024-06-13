from .qBNMC import qBayesNet
from .qBNRejection import qInference

from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag

from qiskit_aer import AerSimulator

from qiskit.quantum_info import Operator

from qiskit_ibm_runtime.ibm_backend import IBMBackend

from pyAgrum import LazyPropagation

import matplotlib.pyplot as plt


class qRuntime:
    """
    Class used to evaluate the thoeretical time of execution of a quantum sampler on a quantum device

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, qinf: qInference, backend: IBMBackend) -> None:
        """
        Initialises the qBaysNet Object 

        Parameters
        ----------
        qinf: qInference
            Quantum rejecetion sampler
        backend: IBMBackend
            Backend to get the execution time on quantum harware

        """
        self.qinf = qinf
        self.default_backend = backend
        self.A_time = None
        self.G_time = None

    def getGateExecutionTime(self, verbose = 0) -> None:
        """
        Stores the execution time of gate A and G
    
        """
        self.A_time = self.getAtime(verbose=verbose)
        self.G_time = self.getGtime(verbose=verbose)

    def getAtime(self, backend: IBMBackend = None, verbose = 0) -> float:
        """
        Estimates the theoredical runtime of the quantum circuit from given backend in seconds

        Parameters
        ---------
        backend: AerSimulator = None
            Backend to transpile the quantum circuit (default set to AerSimulator)

        Returns
        -------
        float
            Estimate of the circuit runtime in seconds

        """
        if backend == None: backend = self.default_backend

        A = self.qinf.getA() #gate depth may be shorter due to optimisation

        circuit = QuantumCircuit(*list(self.qinf.q_registers.values()))
        circuit.compose(A, inplace=True)

        transpiled_circuit = transpile(circuit, backend=backend)

        dag_circuit = circuit_to_dag(transpiled_circuit)
        circuit_depth = dag_circuit.count_ops_longest_path()
        circuit_depth.pop("barrier", None)

        res = 0.0

        for key, val in circuit_depth.items():
            instruction = next(iter(backend.target[key].values()), None) #to be revisited
            res += instruction.duration * val

        if verbose > 0:
            print(f"A gate transpiled circuit depth: {transpiled_circuit.depth()}")    
            print(f"A gate execution time: {res} s")

        return res

    def getGtime(self, backend: IBMBackend = None, verbose = 0) -> float:
        """
        Estimates the theoredical runtime of a Grover iterate from given backend in seconds

        Parameters
        ---------
        backend: AerSimulator = None
            Backend to transpile the quantum circuit (default set to AerSimulator)

        Returns
        -------
        float
            Estimate of the circuit runtime in seconds

        """
        if backend == None: backend = self.default_backend

        evidence_n_id = {self.qinf.qbn.bn.nodeId(self.qinf.qbn.bn.variable(key)): val
                         for key, val in self.qinf.evidence.items()}
        
        evidence_qbs = self.qinf.getEvidenceQuBits(evidence_n_id)

        A = self.qinf.getA() #gate depth may be shorter due to optimisation
        G = self.qinf.getG(A, evidence_qbs=evidence_qbs)

        circuit = QuantumCircuit(*list(self.qinf.q_registers.values()))
        circuit.compose(G, inplace=True)

        transpiled_circuit = transpile(circuit, backend=backend)

        dag_circuit = circuit_to_dag(transpiled_circuit)
        circuit_depth = dag_circuit.count_ops_longest_path()
        circuit_depth.pop("barrier", None)

        res = 0.0

        for key, val in circuit_depth.items():
            instruction = next(iter(backend.target[key].values()), None) #to be revisited
            res += instruction.duration * val
        
        if verbose > 0:
            print(f"A gate transpiled circuit depth: {transpiled_circuit.depth()}")    
            print(f"A gate execution time: {res} s")
    
        return res

    def rejectionSamplingRuntime(self) -> float:
        """
        Uses gate execution time from before to compute the total time of the rejection sampling process 

        Returns
        -------
        float

        """
        if self.A_time is None or self.G_time is None:
            self.getGateExecutionTime()

        res = 0.0
        res += self.qinf.log["A"] * self.A_time
        res += self.qinf.log["G"] * self.G_time
        return res

    def compareInference(self, ie = None, ax=None):
        """
        Compares 2 inference by plotting all the points from qInference and ie
        
        """

        bn = self.qinf.qbn.bn

        if ie is None:
            ie = LazyPropagation(bn)
            ie.setEvidence(self.qinf.evidence)
            ie.makeInference()

        exact=[]
        appro=[]
        errmax=0
        for node in bn.nodes():
            # potentials as list
            print(node, ie.posterior(node).tolist())
            exact += ie.posterior(node).tolist()
            appro += self.qinf.posterior(node).tolist()
            errmax=max(errmax,
                    (ie.posterior(node) - self.qinf.posterior(node)).abs().max())
    
        if errmax < 1e-10: errmax=0
        if ax == None:
            fig = plt.Figure(figsize=(4,4))
            ax = plt.gca() # default axis for plt
        
           
        ax.plot(exact,appro,'ro')
        ax.set_title("{} vs {}\n Max iter {} \nMax error {:2.4} in {:2.4} seconds".format(
            str(type(ie)).split(".")[2].split("_")[0][0:-2], # name of first inference
            str(type(self.qinf)).split(".")[2], # name of second inference
            self.qinf.max_iter,
            errmax,
            self.rejectionSamplingRuntime())
            )