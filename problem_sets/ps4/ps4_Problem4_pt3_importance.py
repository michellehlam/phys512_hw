import numpy as np
import matplotlib.pyplot as plt
import camb

# import previously processed data
# old chain from Problem 3
mcmc = np.loadtxt('chain_chi_10000steps.txt')
chain = mcmc[:,1:7]

# expected params from new chain
mcmc2 = np.loadtxt('chain_chi_tau.txt')
chain2 = mcmc2[:,1:7]
expected_pars = chain2[-1]
npar = len(expected_pars)
print(np.shape(expected_pars))
par_errs = np.zeros(npar)
for i in range(npar):
    par_errs[i] = np.std(chain2[:,i])

def prior_chisq(pars,par_priors,par_errs):
    if par_priors is None:
        return 0
    par_shifts=pars-par_priors
    return np.sum((par_shifts/par_errs)**2)

#importance sample the happy chain
nsamp=chain2.shape[0]
weight=np.zeros(nsamp)
chivec=np.zeros(nsamp)

for i in range(nsamp):
    chisq=prior_chisq(chain[i,:],expected_pars,par_errs)
    chivec[i]=chisq
#    weight[i]=np.exp(-0.5*chisq)

#plt.plot(chivec)
#plt.show()
chivec=chivec-chivec.mean()
weight=np.exp(0.5*chivec)

new_params = np.zeros(npar)
for i in range(len(par_errs)):
    print('importance sampled parameter ',i,' has mean ',np.sum(weight*chain2[:,i])/np.sum(weight))
    new_params[i] = np.sum(weight*chain2[:,i])/np.sum(weight)
print('new_params:', new_params)
def get_spectrum(pars, lmax = 2507): # taken from planck_likelihood.pyre ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt[2:]

raw = np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt')
err = 0.5*(raw[:,2] + raw[:,3])
chisq = np.sum((raw[:,1]-get_spectrum(new_params)[:len(raw[:,1])])**2/err**2)
print('chisq: ', chisq)


print('chain with tau constraint: ', expected_pars)
chisq = np.sum((raw[:,1]-get_spectrum(expected_pars)[:len(raw[:,1])])**2/err**2)
print('chisq: ', chisq)
