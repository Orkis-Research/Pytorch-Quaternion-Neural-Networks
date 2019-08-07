##########################################################
# pytorch-qnn v1.0
# Titouan Parcollet
# LIA, Université d'Avignon et des Pays du Vaucluse
# ORKIS, Aix-en-provence
# October 2018
##########################################################

import numpy             as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib          import rc

# LATEX
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size'   : 15})
rc('text', usetex=True)

label = np.zeros(400)
for i in range(400):
    label[i] = i*5

first  = 10
second = 50
third  = 100

q_acc_f  = np.loadtxt("../out/memory_task_acc_q_"+str(first)+".txt")
q_loss_f = np.loadtxt("../out/memory_task_loss_q_"+str(first)+".txt")
q_acc_s  = np.loadtxt("../out/memory_task_acc_q_"+str(second)+".txt")
q_loss_s = np.loadtxt("../out/memory_task_loss_q_"+str(second)+".txt")
q_acc_t  = np.loadtxt("../out/memory_task_acc_q_"+str(third)+".txt")
q_loss_t = np.loadtxt("../out/memory_task_loss_q_"+str(third)+".txt")

r_acc_f  = np.loadtxt("../out/memory_task_acc_r_"+str(first)+".txt")
r_loss_f = np.loadtxt("../out/memory_task_loss_r_"+str(first)+".txt")
r_acc_s  = np.loadtxt("../out/memory_task_acc_r_"+str(second)+".txt")
r_loss_s = np.loadtxt("../out/memory_task_loss_r_"+str(second)+".txt")
r_acc_t  = np.loadtxt("../out/memory_task_acc_r_"+str(third)+".txt")
r_loss_t = np.loadtxt("../out/memory_task_loss_r_"+str(third)+".txt")


q_acc_f = q_acc_f[:] * 100
r_acc_f = r_acc_f[:] * 100
q_acc_s = q_acc_s[:] * 100
r_acc_s = r_acc_s[:] * 100
q_acc_t = q_acc_t[:] * 100
r_acc_t = r_acc_t[:] * 100

f, axarr = plt.subplots(2, 3,figsize=(8,5.5))

axarr[0, 0].plot(label, q_loss_f, label='QLSTM')
axarr[0, 0].plot(label, r_loss_f, label='LSTM')
axarr[0, 0].set_title('T=10')
axarr[0, 1].plot(label, q_loss_s)
axarr[0, 1].plot(label, r_loss_s)
axarr[0, 1].set_title('T=50')
axarr[1, 0].plot(label, q_acc_f)
axarr[1, 0].plot(label, r_acc_f)
axarr[1, 1].plot(label, q_acc_s)
axarr[1, 1].plot(label, r_acc_s)
axarr[0, 2].plot(label, q_loss_t)
axarr[0, 2].plot(label, r_loss_t)
axarr[0, 2].set_title('T=100')
axarr[1, 2].plot(label, q_acc_t)
axarr[1, 2].plot(label, r_acc_t)

cpt = 0

for i, row in enumerate(axarr):
    for j, cell in enumerate(row):

        if j == 0:
            if i == 0:
                cell.set_ylabel("Cross entropy", labelpad=21)
            else:
                cell.set_ylabel("Accuracy \%")

f.text(0.51, 0.02, 'Epochs', ha='center')

# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
handles, labels = axarr[0,0].get_legend_handles_labels()
f.legend(handles, labels, loc='upper right', prop={'size': 12})
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 2]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

plt.savefig('../out/curves.png', format='png', dpi=1200)
