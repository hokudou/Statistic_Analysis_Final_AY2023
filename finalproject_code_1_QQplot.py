import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

SEEDs = [11, 12, 13]

data1 = np.genfromtxt("data1.csv", delimiter=",")
data2 = np.genfromtxt("data2.csv", delimiter=",")
data3 = np.genfromtxt("data3.csv", delimiter=",")
dataset = [data1, data2, data3]
for SEED in SEEDs:
    np.random.seed(SEED)
    for data in dataset:
        idx = dataset.index(data)
        if idx == 0 or idx == 1:  # for data1 and data2
            locA, scaleA = stats.halfnorm.fit(data, floc=0)
            B, locB, scaleB = stats.lognorm.fit(data, floc=0)
            locC, scaleC = stats.levy.fit(data)
            locD, scaleD = stats.expon.fit(data, floc=0)
            E, locE, scaleE = stats.pareto.fit(data, floc=0)
            F, locF, scaleF = stats.gamma.fit(data, floc=0)
            locG, scaleG = stats.gumbel_l.fit(data)
            locH, scaleH = stats.cauchy.fit(data)
            # estimate each parameter

            # p1A = math.sqrt(math.pi/2)/scaleA
            # p1B = np.log(scaleB)
            # p2B = B/scaleB
            # p2D = 1/scaleD
            # p2E = E
            # p1F = F
            # p2F = 1/scaleF

            y_A = stats.halfnorm.rvs(loc=0, scale=scaleA, size=len(
                data), random_state=SEED)
            y_B = stats.lognorm.rvs(B, loc=0, scale=scaleB, size=len(
                data), random_state=SEED)
            y_C = stats.levy.rvs(loc=locC, scale=scaleC, size=len(
                data), random_state=SEED)
            y_D = stats.expon.rvs(loc=0, scale=scaleD, size=len(
                data), random_state=SEED)
            y_E = stats.pareto.rvs(E, loc=0, scale=scaleE, size=len(
                data), random_state=SEED)
            y_F = stats.gamma.rvs(F, loc=0, scale=scaleF, size=len(
                data), random_state=SEED)
            y_G = stats.gumbel_l.rvs(loc=locG, scale=scaleG, size=len(
                data), random_state=SEED)
            y_H = stats.cauchy.rvs(loc=locH, scale=scaleH, size=len(
                data), random_state=SEED)
            # generate random variables

            data.sort()
            y_A.sort()
            y_B.sort()
            y_C.sort()
            y_D.sort()
            y_E.sort()
            y_F.sort()
            y_G.sort()
            y_H.sort()

            plt.scatter(data, y_A, s=1, label="Half-normal distribution")
            plt.scatter(data, y_B, s=1, label="Lognormal distribution")
            plt.scatter(data, y_C, s=1, label="Lévy distribution")
            plt.scatter(data, y_D, s=1, label="Exponential distribution")
            plt.scatter(data, y_E, s=1, label="Pareto distribution")
            plt.scatter(data, y_F, s=1, label="Gamma distribution")
            plt.scatter(data, y_G, s=1, label="Gumbel distribution")
            plt.scatter(data, y_H, s=1, label="Cauchy distribution")
            # build Q-Q plot
            if (idx == 0):
                plt.xlim((0, 3))
                plt.ylim((0, 3))
            else:
                plt.xlim((2, 8))
                plt.ylim((2, 8))
            legend = plt.legend(markerscale=5)
            plt.axline((0, 0), slope=1, color='#8e8e8e', lw=2)
            plt.xlabel("Sample Quantiles")
            plt.ylabel("Theoretical Quantiles")
            plt.savefig("Q-Qplot " + str(idx+1) + " (SEED="+str(SEED)+").eps")
            plt.clf()

        else:  # for data3
            locC, scaleC = stats.levy.fit(data)
            locG, scaleG = stats.gumbel_l.fit(data)
            locH, scaleH = stats.cauchy.fit(data)
            # estimate each parameter

            # p2D = 1/scaleD

            y_C = stats.levy.rvs(loc=locC, scale=scaleC, size=len(
                data), random_state=SEED)
            y_G = stats.gumbel_l.rvs(loc=locG, scale=scaleG, size=len(
                data), random_state=SEED)
            y_H = stats.cauchy.rvs(loc=locH, scale=scaleH, size=len(
                data), random_state=SEED)
            # generate random variables

            data.sort()
            y_C.sort()
            y_G.sort()
            y_H.sort()

            plt.scatter(data, y_C, s=1, label="Lévy distribution")
            plt.scatter(data, y_G, s=1, label="Gumbel distribution")
            plt.scatter(data, y_H, s=1, label="Cauchy distribution")
            # build Q-Q plot
            plt.xlim((-50, 50))
            plt.ylim((-50, 50))
            legend = plt.legend(markerscale=5)
            plt.axline((0, 0), slope=1, color='#8e8e8e', lw=2)
            plt.xlabel("Sample Quantiles")
            plt.ylabel("Theoretical Quantiles")
            plt.savefig("Q-Qplot " + str(idx+1) + " (SEED="+str(SEED)+").eps")
            plt.clf()
