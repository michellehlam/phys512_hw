import numpy as np 
import matplotlib.pyplot as plt

data = np.loadtxt('chain_chi_10000steps.txt')
endpoint = 10000

plt.plot(data[:,0])
plt.title('chisq')
plt.show()

print('chisq: ', (data[:,0])[-1])

chain = data[:,1:7]
expected_pars = chain[-1]
npar = len(expected_pars)

par_errs = np.zeros(npar)
for i in range(npar):
    par_errs[i] = np.std(chain[:,i])

# calculate dark energy
h = expected_pars[0]/100

ombh2_omch2 = (expected_pars[1] + expected_pars[2])
darkE = 1- ombh2_omch2/h**2
# calc dark energy uncertainty
dh = par_errs[0]/100
dh2 = 2*dh/h
d_ombh2_omch2 = par_errs[2] + par_errs[3] # uncertainty rule for summing
d_om = np.sqrt((d_ombh2_omch2/ombh2_omch2)**2 + (dh2/h**2)**2)*(ombh2_omch2/h**2)

print('dark energy is:', darkE, '+/-', d_om)

labels = ['H0', 'ombh2', 'omch2', 'tau', 'As', 'ns']
print(data.shape[1])
for i in range(1,7):
    plt.loglog(np.abs(np.fft.rfft(data[:,i])))
    plt.title(labels[i-1])
    plt.show()

    dm = np.std(data[:,i], axis=0)
    m = np.mean(data[:,i], axis=0)
    print(labels[i-1], ': ', m, '+/-', dm)


