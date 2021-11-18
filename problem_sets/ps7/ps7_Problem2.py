import numpy as np
import matplotlib.pyplot as plt

#x= np.linspace(0,100, 100)
#k = 0.005
#sigma = k*1e5
#plt.plot(x,np.exp(-k*x), label = 'exponential')
#plt.plot(x,np.exp(-x**2/sigma**2/2), label = 'gauss')
#plt.legend()
#plt.show()


# goal: write rejection method to make exponential deviates from another distribution
# - function for bounding distribution: 
# - assume exponential deviates = non-negative , cut off distributoin at 0 
# Output: histogram of deviates 

# i.e. want to randomly sample exopnential curve  e^-kt


def cdf_lorentz(n):
    q= np.pi*(np.random.rand(n)-0.5)
    return np.tan(q)

n = int(1e6)
t = cdf_lorentz(n) # distribution
y_lor = 1/(1+t**2)*np.random.rand(n) # y_value on distribution, multiplied by 0 to 1, 
#plt.plot(t,y_lor, '.')
#plt.xlim(-10,10)
#plt.show()

bins = np.linspace(0,10,501)
aa, bb = np.histogram(t,bins)
aa = aa/aa.sum()

cents = 0.5*(bins[1:]+bins[:-1])
actual = 1/(1+cents**2)
actual = actual/actual.sum()

#---  plot lorentz ---
#plt.plot(cents,aa, '+')
#plt.plot(cents, actual, label = 'analytical')
#plt.show()


k = 1
lor = 1/(1+cents**2)
myexp = np.exp(-k*cents)

#  ---- plot lorentz vs. actual function you want ----
#plt.plot(cents, lor)
#plt.plot(t,y_lor, '.')
#plt.plot(cents,myexp, label = 'exp')
#plt.xlim(-10,10)
#plt.legend()
#plt.show()

accept = y_lor < np.exp(-k*t)
t_accept = t[accept]
accept = t_accept>=0
t_use = t_accept[accept]

print(np.mean(t_use))
print(np.std(t_use))
print('percent accepted: ', np.mean(accept))

aa, bb = np.histogram(t_use, bins)
aa = aa/aa.sum()

plt.plot(cents,aa, '+', label = 'rejection method')
plt.plot(cents, lor/lor.sum(), label = 'analytical lorentz')
plt.plot(cents, myexp/myexp.sum(), label = 'analytical exponential')
plt.legend()
plt.show()


