import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def boots(n):
    output = np.zeros(n)
    np.random.seed(SEED + i)
    sample_indices1 = np.random.choice(
        len(sample), n, replace=True)
    output = sample[sample_indices1.astype(int)]
    return output


SEED = 1106
np.random.seed(SEED)
original_n = 500000
rap = 10  # the number of repeat
x_record = np.zeros(rap)
t_diff_record = np.zeros(rap)
cauchy_diff_record = np.zeros(rap)
# prepare sample following student-t model
sample = stats.t.rvs(1.05, loc=0.05, scale=1.05, size=original_n)
# sample = stats.cauchy.rvs(loc=0, scale=1, size=original_n)
print("Sample ready")
for i in range(rap):
    # reduce number of parametric sample 50000 each from 5000000
    n = original_n - 50000 * i
    x_record[i] = n

    resample = boots(n)  # the parametric sample

    CDF_N_X = np.sort(resample)
    CDF_N_Y = np.arange(n) / float(n)  # construct cdf
    absdiff_t = np.abs(
        CDF_N_Y - stats.t.cdf(CDF_N_X, 1.05, loc=0.05, scale=1.05))
    absdiff_cauchy = np.abs(
        CDF_N_Y - stats.cauchy.cdf(CDF_N_X, loc=0, scale=1))
    t_diff_record[i] = np.max(absdiff_t)
    # record the biggest difference
    cauchy_diff_record[i] = np.max(absdiff_cauchy)
    print(i+1, "resampling done")
plt.xlabel("Sample size")
plt.ylabel("D")
plt.plot(x_record, t_diff_record, marker="8", label="students t")
plt.plot(x_record, cauchy_diff_record, marker="s", label="cauchy")
plt.legend()
plt.savefig("GCtheorem.eps")
plt.show()
plt.clf()
