import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

np.random.seed(1106)

x = np.arange(0, 10+0.01, 0.01)

t_y = stats.t.pdf(x, 1.05, loc=0.05, scale=1.05)  # student-t model
cauchy_y = stats.cauchy.pdf(x, loc=0, scale=1)  # cauchy model
plt.plot(x, t_y, label='students t')
plt.plot(x, cauchy_y, label='cauchy')
plt.xlabel("input value")
plt.ylabel("pdf value")
plt.legend()
plt.savefig("comparison.eps")
plt.show()
