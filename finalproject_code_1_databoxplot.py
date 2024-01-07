import numpy as np
import matplotlib.pyplot as plt

data1 = np.genfromtxt("data1.csv", delimiter=",")
data2 = np.genfromtxt("data2.csv", delimiter=",")
data3 = np.genfromtxt("data3.csv", delimiter=",")
dataset = [data1, data2, data3]
fig, axs = plt.subplots(3)
for data in dataset:
    idx = dataset.index(data)
    axs[idx].set_title("data"+str(idx+1))
    axs[idx].boxplot(data, 0, 'rs', 0)

fig.subplots_adjust(bottom=0.05, top=0.9, hspace=0.5)
plt.savefig("databox.eps")
