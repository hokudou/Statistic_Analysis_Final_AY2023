import numpy as np
import matplotlib.pyplot as plt

data1 = np.genfromtxt("data1.csv", delimiter=",")
data2 = np.genfromtxt("data2.csv", delimiter=",")
data3 = np.genfromtxt("data3.csv", delimiter=",")
dataset = [data1, data2, data3]
for data in dataset:
    idx = dataset.index(data)
    plt.hist(data)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig("data" + str(idx+1) + ".eps")
    plt.clf()