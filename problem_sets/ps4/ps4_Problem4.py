import numpy as np
import matplotlib.pyplot as plt 

mcmc = np.loadtxt('chain_chi.txt')
chain_H0 = mcmc[:,1] 
chisq = mcmc[:,0]
plt.plot(chisq)
plt.xlabel('nstep')
plt.show()

plt.plot(chain_H0)
plt.title('H0')
plt.show()

plt.plot(mcmc[:,3])
plt.title('tau')
plt.show()
