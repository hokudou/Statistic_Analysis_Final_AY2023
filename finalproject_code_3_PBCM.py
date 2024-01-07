import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def A_GOF(data):
    GOF = 0
    loc, scale = stats.halfnorm.fit(data)
    for i in data:
        GOF += stats.halfnorm.logpdf(i, loc, scale)
    return GOF


def B_GOF(data):
    GOF = 0
    b, loc, scale = stats.gamma.fit(data)
    for i in data:
        GOF += stats.gamma.logpdf(i, b, loc, scale)
    return GOF


M = 1000  # number of parametric sample
SEED = 1106
data = np.genfromtxt("data1.csv", delimiter=",")
A_true_GOF_diff = np.zeros(M)  # A refers to half-normal
B_true_GOF_diff = np.zeros(M)  # B refers to gamma


for i in range(M):
    SEED += 1
    np.random.seed(SEED)
    sample_indices1 = np.random.choice(
        len(data), len(data), replace=True)
    # get non-parametric sample
    nonparaboots = data[sample_indices1.astype(int)]
    locA1, scaleA1 = stats.halfnorm.fit(nonparaboots)
    B1, locB1, scaleB1 = stats.gamma.fit(nonparaboots)
    parabootsA = stats.halfnorm.rvs(locA1, scaleA1, size=len(data))
    # get parametric sample from each
    parabootsB = stats.gamma.rvs(B1, locB1, scaleB1, size=len(data))
    GOD_AtoA = A_GOF(parabootsA)
    GOD_AtoB = B_GOF(parabootsA)  # assess GOF
    A_true_GOF_diff[i] = GOD_AtoA - GOD_AtoB
    GOD_BtoA = A_GOF(parabootsB)
    GOD_BtoB = B_GOF(parabootsB)  # assess GOF
    B_true_GOF_diff[i] = GOD_BtoA - GOD_BtoB

# GOF difference from empirical data
observed_difference = A_GOF(data)-B_GOF(data)
plt.hist(A_true_GOF_diff, bins=30, range=(-30, 30), fill=False,
         label="Halfnormal is true", density=True, edgecolor='red')
plt.hist(B_true_GOF_diff, bins=30, range=(-30, 30), fill=False,
         label="Gamma is true", density=True, edgecolor='blue')
plt.axline(xy1=(observed_difference, 0), xy2=(
    observed_difference, 0.01), color="black", linestyle="--")
plt.xlabel(r"$Log\ell_{Halfnormal} - Log\ell_{Gamma}$")
plt.ylabel("Density")
plt.legend()
plt.savefig("PBCM.eps")
plt.show()
