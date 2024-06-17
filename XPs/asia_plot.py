#parameters

scaling_min = 4
scaling_max = 5.5
num_runs = 100
num_evidence_var = 3
max_iter = 1000

qinf_rt_list = list()
qinf_me_list = list()

mc_rt_list = list()
mc_me_list = list()

ev_prob_list = list()

#imports

import sys
if sys.path[-1] != "..": sys.path.append("..")

from source.qBN.qBNMC import qBayesNet
from source.qBN.qBNRejection import qInference
from XPs.qBNRT import qRuntime

import pyAgrum as gum

import random
import numpy as np
import matplotlib.pyplot as plt

from qiskit_ibm_runtime import QiskitRuntimeService

#backend setup

service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token='1b6910ff55c1d3853e5c8e2ca2b0dbbc3b415fb897d26a6c272c63254527581c824aea1180585f706ab8263318f3c553549d136ca32952ef401abb54011eee33'
)

backend = service.get_backend("ibm_brisbane")

#functions

def getRandomBinaryCPT(num_parents):
    """
    Returns Binary CPT as multi-dimentional list to be added to variable
    """
    if num_parents <= 0:
        r = random.random()
        return [r, 1-r]
    else:
        return [getRandomBinaryCPT(num_parents-1), getRandomBinaryCPT(num_parents-1)]

def randomChoice(elements, num_choice):
    """
    Removes num_choice elements from the elements set at random and returns them in a set 
    """
    res = set()
    for i in range(num_choice):
        chosen = random.choice(list(elements))
        elements.discard(chosen)
        res.add(chosen)
    return res

def modifyBinaryCPT(cpt, state, scaling):
    """
    Returns Binary CPT as multi-dimentional list with the state probability divided by scaling
    """
    if len(cpt.shape) == 1:
        if state == 0:
            return [cpt[0]/scaling, 1-cpt[0]/scaling]
        else:
            return [1-cpt[1]/scaling, cpt[1]/scaling]
    else:
        return [modifyBinaryCPT(cpt[0], state, scaling), modifyBinaryCPT(cpt[1], state, scaling)]

#Bayeset setup

asia_bn = gum.loadBN("asia.bif")
qbn = qBayesNet(asia_bn)
qc = qbn.buildCircuit(add_measure=True)

#ploting

for i in range(num_runs):

    #Randomly Chosen Evidence and Target
    n_ids = asia_bn.nodes()
    evidence = {ev_id: random.randint(0,1) for ev_id in randomChoice(n_ids, num_evidence_var)}
    target = list(randomChoice(n_ids, 1))[0]

    #modifying the probabilities
    scaling = random.random()*(scaling_max - scaling_min) + scaling_min
    for n_id, n_state in evidence.items():
        asia_bn.cpt(n_id)[:] = modifyBinaryCPT(asia_bn.cpt(n_id), n_state, scaling)
    old_target_cpt = asia_bn.cpt(target).tolist()
    asia_bn.cpt(target).translate(1e-5).normalizeAsCPT()

    #Lazy Propagation Benchmark
    ie = gum.LazyPropagation(asia_bn)
    ie.setEvidence(evidence)
    ie.makeInference()
    print(f"Evidence: {evidence}, Target Node: {target}")
    print(f"Evidence probability: {ie.evidenceProbability()}")
    ev_prob_list.append(ie.evidenceProbability())


    #Monte Carlo Classical Rejection Sampling
    mc = gum.MonteCarloSampling(asia_bn)
    mc.setEpsilon(1e-20)
    mc.setMaxTime(1e20)
    mc.setEvidence(evidence)
    mc.setMaxIter(max_iter)
    mc.makeInference()
    mc_run_time = mc.currentTime()
    mc_max_error = (mc.posterior(target).toarray() - ie.posterior(target).toarray()).max()
    print(f"\tMC - Run time: {mc_run_time}, Max Error: {mc_max_error}")
    mc_rt_list.append(mc_run_time)
    mc_me_list.append(mc_max_error)

    #Quantum Rejection Sampling
    qinf = qInference(qbn)
    qrt = qRuntime(qinf, backend)
    qinf.setEvidence(evidence)
    qinf.setMaxIter(max_iter)
    qinf.makeInference(verbose=0)
    qinf_run_time = qrt.rejectionSamplingRuntime()
    qinf_max_error = (qinf.posterior(target).toarray() - ie.posterior(target).toarray()).max()
    print(f"\tQS - Run time: {qinf_run_time}, Max Error: {qinf_max_error}")
    qinf_rt_list.append(qinf_run_time)
    qinf_me_list.append(qinf_max_error)

    #resetting the probabilities
    for n_id, n_state in evidence.items():
        asia_bn.cpt(n_id)[:] = modifyBinaryCPT(asia_bn.cpt(n_id), n_state, 1/scaling)
    asia_bn.cpt(target)[:] = old_target_cpt

#saving the output
with open("asia_output.txt", "w") as output:
    output.write(f"scaling: {scaling}, num_runs: {num_runs}, num_evidence_var: {num_evidence_var}, max_iter: {max_iter}\n")
    output.write("ev_prob_list: \n")
    output.write(str(ev_prob_list)+"\n")
    output.write("qinf_rt_list: \n")
    output.write(str(qinf_rt_list)+"\n")
    output.write("mc_rt_list: \n")
    output.write(str(mc_rt_list)+"\n")
    output.write("qinf_me_list: \n")
    output.write(str(qinf_me_list)+"\n")
    output.write("mc_me_list: \n")
    output.write(str(mc_me_list))