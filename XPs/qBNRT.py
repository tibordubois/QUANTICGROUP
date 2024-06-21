import sys
if sys.path[-1] != "..": sys.path.append("..")

from source.qBN.qBNRejection import qInference

from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag

from qiskit_ibm_runtime.ibm_backend import IBMBackend


class qRuntime:
    """
    Class to evaluate the thoeretical execution time of quantum sampler 
    on a quantum backend

    Attributes
    ----------

    qinf: qInference
        qInference Object from qBNRejection

    default_backend: IBMBackend
        Default backend to gather gate execution times

    A_time: float
        Gate A execution time in seconds

    G_time: float
        Gate A execution time in seconds

    Methods
    -------

    getGateExecutionTime(self, verbose: int = 0) -> None:
        Stores the execution time of gate A and G

    getAtime(self, backend: IBMBackend = None, verbose: int = 0) -> float:
        Estimates the theoredical runtime of the quantum circuit from given backend 
        in seconds

    getGtime(self, backend: IBMBackend = None, verbose: int = 0) -> float:
        Estimates the theoredical runtime of a Grover iterate from given backend 
        in seconds

    rejectionSamplingRuntime(self) -> float:
        Uses gate execution time from before to compute the total time of the 
        rejection sampling process 

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

    def getGateExecutionTime(self, verbose: int = 0) -> None:
        """
        Stores the execution time of gate A and G
    
        Parameters
        ----------
        verbose: int = 0
            Verbose

        """
        self.A_time = self.getAtime(verbose=verbose)
        self.G_time = self.getGtime(verbose=verbose)

    def getAtime(self, backend: IBMBackend = None, verbose: int = 0) -> float:
        """
        Estimates the theoredical runtime of the quantum circuit from given backend 
        in seconds

        Parameters
        ----------
        backend: IBMBackend = None
            Backend to transpile the quantum circuit (default set to AerSimulator)
        verbose: int = 0
            Verbose

        Returns
        -------
        float
            Estimate of the circuit runtime in seconds

        """
    
        if backend == None: backend = self.default_backend

        circuit  = QuantumCircuit(*list(self.qinf.q_registers.values()))
        self.qinf.addA(circuit) #gate depth may be shorter due to optimisation

        transpiled_circuit = transpile(circuit, backend=backend)

        dag_circuit = circuit_to_dag(transpiled_circuit)
        circuit_depth = dag_circuit.count_ops_longest_path()
        circuit_depth.pop("barrier", None)

        res = 0.0

        for key, val in circuit_depth.items():
            instruction = next(iter(backend.target[key].values()), None) 
            res += instruction.duration * val

        if verbose > 0:
            print(f"A gate transpiled circuit depth: {transpiled_circuit.depth()}")    
            print(f"A gate execution time: {res} s")

        return res

    def getGtime(self, backend: IBMBackend = None, verbose: int = 0) -> float:
        """
        Estimates the theoredical runtime of a Grover iterate from given backend 
        in seconds

        Parameters
        ----------
        backend: AerSimulator = None
            Backend to transpile the quantum circuit (default set to AerSimulator)
        verbose: int = 0
            Verbose

        Returns
        -------
        float
            Estimate of the circuit runtime in seconds

        """
        if backend == None: backend = self.default_backend

        evidence_n_id = {self.qinf.qbn.bn.nodeId(self.qinf.qbn.bn.variable(key)): val
                         for key, val in self.qinf.evidence.items()}

        evidence_qbs = self.qinf.getEvidenceQuBits(evidence_n_id)

        A  = QuantumCircuit(*list(self.qinf.q_registers.values()))
        self.qinf.addA(A)

        circuit = QuantumCircuit(*list(self.qinf.q_registers.values()))
        self.qinf.addG(circuit, A, evidence_qbs)

        transpiled_circuit = transpile(circuit, backend=backend)

        dag_circuit = circuit_to_dag(transpiled_circuit)
        circuit_depth = dag_circuit.count_ops_longest_path()
        circuit_depth.pop("barrier", None)

        res = 0.0

        for key, val in circuit_depth.items():
            instruction = next(iter(backend.target[key].values()), None) 
            res += instruction.duration * val
        
        if verbose > 0:
            print(f"A gate transpiled circuit depth: {transpiled_circuit.depth()}")    
            print(f"A gate execution time: {res} s")
    
        return res

    def rejectionSamplingRuntime(self) -> float:
        """
        Uses gate execution time from before to compute the total time of the 
        rejection sampling process 

        Returns
        -------
        float
            Estimate of the circuit runtime in seconds

        """
        if self.A_time is None or self.G_time is None:
            self.getGateExecutionTime()

        res = 0.0
        res += self.qinf.log["A"] * self.A_time
        res += self.qinf.log["G"] * self.G_time
        return res
