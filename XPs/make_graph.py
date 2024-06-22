#parameters

num_runs = 0
max_iter = 0

scaling_min = 4.5
scaling_max = 5.2
num_runs = 100
num_evidence_var = 3
max_iter = 100

ev_prob_list = list()
mc_rt_list = list()
mc_me_list = list()
qinf_rt_list = list()
qinf_me_list = list()

prediction_range = list()
prediction_mc = list()
prediction_inf = list()

#imports

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

#parsing

with open("asia_output.txt") as f:

    val = f.readline().strip().split(' ')
    scaling_min = val[1]
    val = f.readline().strip().split(' ')
    scaling_max = val[1]
    val = f.readline().strip().split(' ')
    num_runs = val[1]
    val = f.readline().strip().split(' ')
    num_evidence_var = val[1]
    val = f.readline().strip().split(' ')
    max_iter = val[1]

    val = f.readline().strip().split(' ')

    for line in f:
        val = line.strip().split(' ')
        ev_prob_list.append(float(val[0]))
        mc_rt_list.append(float(val[1]))
        mc_me_list.append(float(val[2]))
        qinf_rt_list.append(float(val[3]))
        qinf_me_list.append(float(val[4]))

#prediction

v_log_func = np.vectorize(lambda x: np.log10(x))
v_exp_func = np.vectorize(lambda x: np.power(10, x))

log_proba = v_log_func(ev_prob_list)
log_mc_rt = v_log_func(mc_rt_list)
log_qinf_rt = v_log_func(qinf_rt_list)

log_prediction_range = np.linspace(np.log10(min(ev_prob_list)), np.log10(max(ev_prob_list)), num=100, endpoint=True).reshape(-1, 1)

reg_mc = LinearRegression().fit(np.array(log_proba).reshape(-1, 1), np.array(log_mc_rt))
log_prediction_mc = reg_mc.predict(log_prediction_range)

reg_qinf = LinearRegression().fit(np.array(log_proba).reshape(-1, 1), np.array(log_qinf_rt))
log_prediction_qinf = reg_qinf.predict(log_prediction_range)

prediction_mc = v_exp_func(log_prediction_mc)
prediction_inf = v_exp_func(log_prediction_qinf)
prediction_range = v_exp_func(log_prediction_range)

#plotting

#run time

plt.figure(0)

plt.scatter(ev_prob_list, mc_rt_list, color="tab:orange", s=10, label="MC run time")
plt.plot(prediction_range, prediction_mc, color="tab:orange", label="MC linear reg. on 1/x")
plt.scatter(ev_prob_list, qinf_rt_list, color="tab:blue", s=10, label="QI run time")
plt.plot(prediction_range, prediction_inf, color="tab:blue", label="QI linear reg. on 1/x\u00B2")

plt.yscale('log')
plt.xscale('log')
plt.grid(True)
plt.xlabel('Evidence probability')
plt.ylabel('Run time')
plt.title(f'Run time evolution of MC/QI samplers ({num_runs} observations, {max_iter} iterations)')
plt.legend()

plt.savefig('../XPs/plots/TimeProbScatter2.png')

#max error

plt.figure(1)

plt.scatter(ev_prob_list, mc_me_list, color="tab:orange", s=10, label="MC run time")
plt.scatter(ev_prob_list, qinf_me_list, color="tab:blue", s=10, label="QI run time")

plt.yscale('log')
plt.xscale('log')
plt.grid(True)
plt.xlabel('Evidence probability')
plt.ylabel('Max Error')
plt.title(f'Max Error evolution of MC/QI samplers ({num_runs} observations, {max_iter} iterations)')
plt.legend()

plt.savefig('../XPs/plots/ErrProbScatter2.png')