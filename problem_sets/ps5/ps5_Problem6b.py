import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit

# do random walk
n = 1000
y = np.cumsum(np.random.randn(n))

# take fft and square it to get psf
ps = np.abs(np.fft.fft(y))**2
# since last part of psf goes up, avoid in fitting by fitting the earlier points
ps_test = (np.abs(np.fft.fft(y))[:int(len(ps)*0.75)])**2 
n = len(ps_test)
x = np.linspace(1,n+1, n)

# fit to function that's proportional to 1/k^2
def func(x,a):
    return a*x**(-2)

popt, pcov = curve_fit(func, x, ps_test, p0 = [n**2])
print(popt)

# plot psf with our fit
plt.loglog(ps, label = 'psf of random walk')
plt.loglog(func(x,popt), label = '~k^-2 fit')#n**2/x**2)
plt.legend()
plt.show()