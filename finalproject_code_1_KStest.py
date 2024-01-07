import numpy as np
import scipy.stats as stats

data1 = np.genfromtxt("data1.csv", delimiter=",")
data2 = np.genfromtxt("data2.csv", delimiter=",")
data3 = np.genfromtxt("data3.csv", delimiter=",")
dataset = [data1, data2, data3]
for data in dataset:
    idx = dataset.index(data)
    if idx == 0:  # for data1
        locA, scaleA = stats.halfnorm.fit(data, floc=0)
        locD, scaleD = stats.expon.fit(data, floc=0)
        F, locF, scaleF = stats.gamma.fit(data, floc=0)
        locG, scaleG = stats.gumbel_l.fit(data)
        locH, scaleH = stats.cauchy.fit(data)
        Apv = stats.kstest(data, stats.halfnorm(loc=0, scale=scaleA).cdf)[1]
        Dpv = stats.kstest(data, stats.expon(loc=0, scale=scaleD).cdf)[1]
        Fpv = stats.kstest(data, stats.gamma(F, loc=0, scale=scaleF).cdf)[1]
        Gpv = stats.kstest(data, stats.gumbel_l(loc=locG, scale=scaleG).cdf)[1]
        Hpv = stats.kstest(data, stats.cauchy(loc=locH, scale=scaleH).cdf)[1]
        print('[A] p-value:' + str(round(Apv, 4)))
        print('[D] p-value:' + str(round(Dpv, 4)))
        print('[F] p-value:' + str(round(Fpv, 4)))
        print('[G] p-value:' + str(round(Gpv, 4)))
        print('[H] p-value:' + str(round(Hpv, 4)))
    elif idx == 1:  # for data2
        B, locB, scaleB = stats.lognorm.fit(data, floc=0)
        locC, scaleC = stats.levy.fit(data)
        E, locE, scaleE = stats.pareto.fit(data, floc=0)
        F, locF, scaleF = stats.gamma.fit(data, floc=0)
        Bpv = stats.kstest(data, stats.lognorm(B, loc=0, scale=scaleB).cdf)[1]
        Cpv = stats.kstest(data, stats.levy(loc=locC, scale=scaleC).cdf)[1]
        Epv = stats.kstest(data, stats.pareto(E, loc=0, scale=scaleE).cdf)[1]
        Fpv = stats.kstest(data, stats.gamma(F, loc=0, scale=scaleF).cdf)[1]
        print('[B] p-value:' + str(round(Bpv, 4)))
        print('[C] p-value:' + str(round(Cpv, 4)))
        print('[E] p-value:' + str(round(Epv, 4)))
        print('[F] p-value:' + str(round(Fpv, 4)))
    else:  # for data3
        locH, scaleH = stats.cauchy.fit(data)
        Hpv = stats.kstest(data, stats.cauchy(loc=locH, scale=scaleH).cdf)[1]
        print('[H] p-value:' + str(round(Hpv, 4)))
    print("------------")
