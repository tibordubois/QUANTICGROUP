from .qBNMC import qBayesNet
from .qBNInference import qInference

from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag

from qiskit_aer import AerSimulator

from qiskit.quantum_info import Operator

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.ibm_backend import IBMBackend

from pyAgrum import LazyPropagation

import matplotlib.pyplot as plt

service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token='2871fe563fabd1df569acfc900f554cd6c9874c98f9a179d17143ce66570130cde317ac0c002524befd1af86259cb6cdf67e88512c054f44a94e3e948557e249'
)

default_backend = service.get_backend("ibm_brisbane")

# Or save your credentials on disk.
# QiskitRuntimeService.save_account(channel='ibm_quantum', instance='ibm-q/open/main', token='2871fe563fabd1df569acfc900f554cd6c9874c98f9a179d17143ce66570130cde317ac0c002524befd1af86259cb6cdf67e88512c054f44a94e3e948557e249')

class qRuntime:
    """
    """

    def __init__(self, qinf: qInference, backend: IBMBackend = None) -> None:
        """
        """
        self.qinf = qinf

        self.default_backend = default_backend if backend is None \
                                              else backend
        
        self.A_time = None
        self.G_time = None

    def getGateExecutionTime(self) -> None:
        """
        """
        self.A_time = self.samplePrepEstimation()
        self.G_time = self.groverIterateEstimation()

    def samplePrepEstimation(self, backend: IBMBackend = None) -> float:
        """Estimates the theoredical runtime of the quantum circuit from given backend in 
        and delays in seconds

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
        print(transpiled_circuit.depth())

        dag_circuit = circuit_to_dag(transpiled_circuit)
        circuit_depth = dag_circuit.count_ops_longest_path()
        circuit_depth.pop("barrier", None)

        res = 0.0

        for key, val in circuit_depth.items():
            instruction = next(iter(backend.target[key].values()), None) #to be revisited
            res += instruction.duration * val
        print(res)
        return res

    def groverIterateEstimation(self, backend: IBMBackend = None) -> float:
        """
        """
        if backend == None: backend = self.default_backend

        A = self.qinf.getA() #gate depth may be shorter due to optimisation
        G = self.qinf.getG(A, evidence_qbs=self.qinf.evidence)

        circuit = QuantumCircuit(*list(self.qinf.q_registers.values()))
        circuit.compose(G, inplace=True)

        transpiled_circuit = transpile(circuit, backend=backend)
        print(transpiled_circuit.depth())
        dag_circuit = circuit_to_dag(transpiled_circuit)
        circuit_depth = dag_circuit.count_ops_longest_path()
        circuit_depth.pop("barrier", None)

        res = 0.0

        for key, val in circuit_depth.items():
            instruction = next(iter(backend.target[key].values()), None) #to be revisited
            res += instruction.duration * val
        print(res)
        return res

    def rejectionSamplingRuntime(self) -> float:
        """
        """
        if self.A_time is None or self.G_time is None:
            self.getGateExecutionTime()

        res = 0.0
        res += self.qinf.log["A"] * self.A_time
        res += self.qinf.log["G"] * self.G_time
        return res

    def compareInference(self, ie = None, ax=None):
        """
        compare 2 inference by plotting all the points from (posterior(ie),posterior(ie2))
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