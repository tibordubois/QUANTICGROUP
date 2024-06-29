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
from scipy.stats import pearsonr
import numpy as np

#parsing
# restore the filename from command line
import sys
if len(sys.argv)>1:
  filenames = sys.argv[1:]
else:
  filenames = ["asia_output.txt"]

num_runs = 0

for filename in filenames:
  with open(filename) as f:
      val = f.readline().strip().split(' ')
      scaling_min = val[1]
      val = f.readline().strip().split(' ')
      scaling_max = val[1]
      val = f.readline().strip().split(' ')
      #num_runs = int(val[1])
      val = f.readline().strip().split(' ')
      num_evidence_var = val[1]
      val = f.readline().strip().split(' ')
      max_iter = val[1]

      val = f.readline().strip().split(' ')

      for line in f:
          num_runs += 1
          val = line.strip().split(' ')
          ev_prob_list.append(float(val[0]))
          mc_rt_list.append(float(val[1]))
          mc_me_list.append(float(val[2]))
          qinf_rt_list.append(float(val[3]))
          qinf_me_list.append(float(val[4]))

filename = filenames[0].split('.')[0]

#prediction

v_log_func = np.vectorize(lambda x: np.log10(x))
v_exp_func = np.vectorize(lambda x: np.power(10, x))

log_proba = np.array(v_log_func(ev_prob_list)).reshape(-1, 1)
log_mc_rt = np.array(v_log_func(mc_rt_list))
log_qinf_rt = np.array(v_log_func(qinf_rt_list))

log_prediction_range = np.linspace(np.log10(min(ev_prob_list)), np.log10(max(ev_prob_list)), num=100, endpoint=True).reshape(-1, 1)

reg_mc = LinearRegression().fit(log_proba, log_mc_rt)
log_prediction_mc = reg_mc.predict(log_prediction_range)

reg_qinf = LinearRegression().fit(log_proba, log_qinf_rt)
log_prediction_qinf = reg_qinf.predict(log_prediction_range)

prediction_mc = v_exp_func(log_prediction_mc)
prediction_inf = v_exp_func(log_prediction_qinf)
prediction_range = v_exp_func(log_prediction_range)

#plotting

n = len(ev_prob_list)

mc_estimation = np.vectorize(lambda x: np.power(x, reg_mc.coef_)*np.power(10, reg_mc.intercept_))
qinf_estimation = np.vectorize(lambda x: np.power(x, reg_qinf.coef_)*np.power(10, reg_qinf.intercept_))

mc_rmse = np.sqrt(np.array([np.power(mc_estimation(ev_prob_list[i])[0] - mc_rt_list[i], 2) for i in range(n)]).sum(axis=0)/n)
qinf_rmse = np.sqrt(np.array([np.power(qinf_estimation(ev_prob_list[i])[0] - qinf_rt_list[i], 2) for i in range(len(ev_prob_list))]).sum(axis=0)/n)

r_mc = pearsonr(np.hstack(log_proba), log_mc_rt)[0]
r_qinf = pearsonr(np.hstack(log_proba), log_qinf_rt)[0]

#run time

#log-log scale

plt.figure(num=0, figsize=(9,7))

plt.scatter(ev_prob_list, mc_rt_list, color="tab:orange", s=10, label="MC run time")
plt.plot(prediction_range, prediction_mc, color="tab:orange", label=f"MC pred. = 10^{(reg_mc.intercept_):.2f}/x^{-reg_mc.coef_[0]:.2f}, r={r_mc:.2f}")
plt.scatter(ev_prob_list, qinf_rt_list, color="tab:blue", s=10, label="QI run time")
plt.plot(prediction_range, prediction_inf, color="tab:blue", label=f"QI pred. = 10^{(reg_qinf.intercept_):.2f}/x^{-reg_qinf.coef_[0]:.2f}, r={r_qinf:.2f}")

plt.yscale('log')
plt.xscale('log')
plt.grid(visible=True, which='both')
plt.xlabel('Evidence probability')
plt.ylabel('Run time (in seconds)')
plt.title(f'Run time evolution of MC/QI samplers ({num_runs} observations, {max_iter} iterations)\nLinear Regression in log-log scale')
plt.legend()

plt.savefig(f'../XPs/plots/Time-{filename}-loglog.png')

#simple-log scale

plt.figure(num=1, figsize=(9,7))

plt.scatter(ev_prob_list, mc_rt_list, color="tab:orange", s=10, label="MC run time")
plt.plot(prediction_range, prediction_mc, color="tab:orange", label=f"MC pred. = 10^{(reg_mc.intercept_):.2f}/x^{-reg_mc.coef_[0]:.2f}, r={r_mc:.2f}")
plt.scatter(ev_prob_list, qinf_rt_list, color="tab:blue", s=10, label="QI run time")
plt.plot(prediction_range, prediction_inf, color="tab:blue", label=f"QI pred. = 10^{(reg_qinf.intercept_):.2f}/x^{-reg_qinf.coef_[0]:.2f}, r={r_qinf:.2f}")

plt.xscale('log')
plt.grid(visible=True, which='both')
plt.xlabel('Evidence probability')
plt.ylabel('Run time (in seconds)')
plt.title(f'Run time evolution of MC/QI samplers ({num_runs} observations, {max_iter} iterations)\nLinear Regression in log-log scale')
plt.legend()

plt.savefig(f'../XPs/plots/Time-{filename}.png')

#max error

plt.figure(num=2, figsize=(9,7))

plt.scatter(ev_prob_list, mc_me_list, color="tab:orange", s=10, label="MC max error")
plt.scatter(ev_prob_list, qinf_me_list, color="tab:blue", s=10, label="QI max error")

plt.yscale('log')
plt.xscale('log')
plt.grid(visible=True, which='both')
plt.xlabel('Evidence probability')
plt.ylabel('Max Error')
plt.title(f'Max Error evolution of MC/QI samplers ({num_runs} observations, {max_iter} iterations)')
plt.legend()

plt.savefig(f'../XPs/plots/Err-{filename}.png')