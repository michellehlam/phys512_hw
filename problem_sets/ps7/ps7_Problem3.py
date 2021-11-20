import numpy as np
import matplotlib.pyplot as plt

# generate exponential deviates using ratio-of-uniforms 
# - (u,v) plane

# define exponential
def exp_pdf(x, gamma=1):
    return np.exp(-x*gamma)

# define u and v, wrt to pdf
n = int(1e6)
xs = np.linspace(0, 0.5e3, n)
u = np.sqrt(exp_pdf(xs))# u(x)
v = xs*u # 

# get min and max , to find our bounds
minu = np.min(u)
maxu = np.max(u)
minv = np.min(v)
maxv = np.max(v)

# sample random numbers within our bounds
u = np.random.uniform(low = minu, high = maxu, size = n)
v = np.random.uniform(low=minv,high=maxv, size = n)
keepers = u < np.sqrt(exp_pdf(v/u))
# take ratio v/u
rand_exp = v[keepers]/u[keepers]

# plot histogram
plt.hist(rand_exp[np.abs(rand_exp)<100], bins = 1000, density=True, label = 'samples')
xs = np.linspace(0,10,1000)
plt.plot(xs, exp_pdf(xs), label = 'analytical')
plt.xlim(min(xs), max(xs))
plt.legend()
plt.show()

# efficiency
print('percent accepted(efficiency): ', len(rand_exp)/len(u)*100, '%')

