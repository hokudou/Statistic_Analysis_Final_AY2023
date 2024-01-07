import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.lines import Line2D


def para_estimate(data):
    if (idx == 0):
        loc, scale = stats.halfnorm.fit(data, floc=0)
        best_para = np.sqrt(np.pi/2)/scale
    if (idx == 1):
        b, loc, scale = stats.pareto.fit(data, floc=0)
        best_para = scale
    if (idx == 2):
        b, loc, scale = stats.pareto.fit(data, floc=0)
        best_para = b
    if (idx == 3):
        loc, scale = stats.cauchy.fit(data)
        best_para = loc
    if (idx == 4):
        loc, scale = stats.cauchy.fit(data)
        best_para = scale
    return best_para


SEED = 1106
m = 1000  # Number of generated samples in bootstrap and jackknife
alpha = 0.05  # Alpha value to calculate the CI
step = 0.0001  # Specification of interval to convert cdf to inverse cdf
data1 = np.genfromtxt("data1.csv", delimiter=",")
data2 = np.genfromtxt("data2.csv", delimiter=",")
data3 = np.genfromtxt("data3.csv", delimiter=",")
dataset = [data1, data2, data2, data3, data3]
idx = 0
para_name = [r"Bootstrap parameter ${\theta}$ for data1", "Bootstrap parameter k for data2",
             r"Bootstrap parameter ${\alpha}$ for data2", "Bootstrap parameter a for data3", "Bootstrap parameter b for data3"]
# theta(1), k(2), alpha(2), a(3), b(3)
for data in dataset:
    data = (data,)
    package_BCa = stats.bootstrap(data, para_estimate,
                                  n_resamples=m,
                                  method='bca', confidence_level=1 - alpha)
    package_BCa_CI_low = package_BCa.confidence_interval.low
    package_BCa_CI_high = package_BCa.confidence_interval.high
    print(f"The CI with the BCa method by scipy is <{0},{1}>".format(
        round(package_BCa_CI_high, 6), round(package_BCa_CI_low, 6)))
    para = package_BCa.bootstrap_distribution
    empirical_para = para_estimate(data)
    if (idx == 1):
        plt.hist(para, density=False, bins=10)
    else:
        plt.hist(para, density=False, bins=50)
    plt.xlabel(para_name[idx])
    plt.ylabel("Frequency")
    plt.axline(xy1=(package_BCa_CI_low, 0), xy2=(
        package_BCa_CI_low, 2), color="red", linestyle="--")  # draw BCa CI
    plt.axline(xy1=(package_BCa_CI_high, 0), xy2=(
        package_BCa_CI_high, 2), color="red", linestyle="--")  # draw BCa CI
    plt.axline(
        xy1=(empirical_para, 0),
        xy2=(empirical_para, 2),
        color="blue",
        linestyle="--",
    )  # draw empirical parameter
    legend_elements = [
        Line2D([0], [0], marker="_", color="red", label="BCa", linestyle="--"),
        Line2D([0], [0], marker="_", color="blue",
               label="Empirical", linestyle="--"),
    ]
    plt.legend(handles=legend_elements, loc="upper left")
    if (idx == 0):  # theta(1)
        plt.xlim([1.2, 2.2])
    if (idx == 1):  # k(2)
        plt.xlim([2, 2.03])
    if (idx == 2):  # alpha(2)
        plt.xlim([-0.5, 3.5])
    if (idx == 3):  # a(3)
        plt.xlim([-4, 3])
    if (idx == 4):  # b(3)
        plt.xlim([16, 23])
    plt.savefig("parameter_estimation_" + str(idx) + ".eps")
    plt.clf()
    idx += 1
