import unittest
import os
import sys
sys.path.insert(0, os.path.abspath('../'))
from unittest.mock import patch, MagicMock
import pyAgrum as gum
from qBN.qBNRejection import qBNRejection
from qBN.qBNMC import qBNMC
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import RYGate, XGate
import numpy as np



class TestQBayesNetInf(unittest.TestCase):
    
    def test_addA(self):
        bn = gum.fastBN("A->B<-C",2)
        qbn1=qBNMC(bn)
        circ1=qBNMC.buildCircuit(qbn1)
        expected_size=17                                
        self.assertEqual(expected_size,circ1.size())
        inf1=qBNRejection(qbn1)
        inf1.addA(circ1)                                 
        expected_size+=14;                              
        self.assertEqual(expected_size,circ1.size())
    
    def test_addB(self):
        bn = gum.fastBN("A->B<-C",2)
        qbn1=qBNMC(bn)
        circ1=qBNMC.buildCircuit(qbn1)
        expected_size=17
        self.assertEqual(expected_size,circ1.size())
        inf1=qBNRejection(qbn1)
        #print(inf1.all_qbits)
        inf1.evidence = {1 : 0}
        inf1.addB(circ1, inf1.evidence)
        expected_size+=1
        self.assertEqual(expected_size,circ1.size())
    
    def test_addInverse(self):
        bn = gum.fastBN("A->B<-C",2)
        qbn1=qBNMC(bn)
        circ1=qBNMC.buildCircuit(qbn1)
        expected_size=17
        self.assertEqual(expected_size,circ1.size())
        inf1=qBNRejection(qbn1)
        inf1.A = QuantumCircuit(*list(inf1.q_registers.values()))
        inf1.addA(inf1.A)
        inf1.addInverse(circ1,inf1.A)
        expected_size+=14
        self.assertEqual(expected_size,circ1.size())
    
    def test_addZ(self):
        bn = gum.fastBN("A->B<-C",2)
        qbn1=qBNMC(bn)
        circ1=qBNMC.buildCircuit(qbn1)
        expected_size=17
        self.assertEqual(expected_size,circ1.size())
        inf1=qBNRejection(qbn1)
        inf1.evidence = {1 : 0}
        inf1.addZ(circ1, inf1.evidence)
        expected_size+=1;
        self.assertEqual(expected_size,circ1.size())
    
    def test_addS(self):
        bn = gum.fastBN("A->B<-C",2)
        qbn1=qBNMC(bn)
        circ1=qBNMC.buildCircuit(qbn1)
        expected_size=17
        self.assertEqual(expected_size,circ1.size())
        inf1=qBNRejection(qbn1)
        inf1.evidence = {1 : 0}
        inf1.addS(circ1, inf1.evidence)
        expected_size+=3;
        self.assertEqual(expected_size,circ1.size())
    
    def test_makeInference(self):
        bn = gum.fastBN("A->B<-C",2)
        qbn1 = qBNMC(bn)
        ev1 = {"B": 0}
        ie1=gum.LazyPropagation(bn)
        ie1.setEvidence(ev1)
        ie1.makeInference()
        qinf1 = qBNRejection(qbn1)
        qinf1.setEvidence(ev1)
        qinf1.setMaxIter(1000)
        a = ie1.posterior("A")
        qinf1.makeInference()
        b = qinf1.posterior("A")
        print("Heuristic Exacte :")
        print(a)
        print("QI :")
        print(b)
        epsilon = 0.020
        self.assertLess(abs(a[0]-b[0]), epsilon)
        self.assertLess(abs(a[1]-b[1]), epsilon)

if __name__ == '__main__':
    unittest.main()