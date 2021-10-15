import numpy as np 
import camb
# mcmc

# ----------------------------------------------------------------------
#                           Functions
#-----------------------------------------------------------------------

# stepsize: use cholesky, sample from covariance matrix
def get_step(cov):
    chol_step = np.linalg.cholesky(cov)
    step = np.dot(np.random.randn(len(chol_step)),chol_step)
    return step

def get_chisq(y, y_pred, sigma):
    r = y-y_pred[:len(y)]
    chisq = np.sum(r**2/sigma**2)
    return chisq

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

#----------------------------------------------------------------------
# load data 
data = np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt')
err = 0.5*(data[:,2] + data[:,3])
ell = data[:,1]
spec = data[:,2]

# draw trial steps from curvature matrix
fit_error = np.loadtxt('planck_fit_params.txt', usecols = 1)
print(fit_error)
# initial parameters
pars = np.asarray([69, 0.022, 0.12, 0.06, 2.1e-9,0.95])
npars = len(pars) # number of params

# take covariant steps, so covariance:
covar = np.diag(fit_error)

# Choose number of steps to take
nstep = 10
# keep track of chisq as we walk around in param space
chisqs = np.zeros(nstep)
chi_cur = get_chisq(spec, get_spectrum(pars),err)
chain = np.zeros([nstep, npars])
npass = 0

for i in range(nstep):
    dpars = get_step(np.diag(np.diag(covar)))
    trial_pars = pars + dpars
    print(dpars)
    print(trial_pars)
    if trial_pars[3]<=0:
        pass
    else:
        y_new = get_spectrum(trial_pars)
        trial_chisq = get_chisq(spec, y_new, err)
        del_chisq = trial_chisq - chisqs[-1]
        prob_step = np.exp(-0.5*del_chisq)

        if np.random.rand(1)<prob_step:
            pars = trial_pars
            chi_cur= trial_chisq
            npass+=1
    chain[i,:]=pars
    chisqs[i] = chi_cur
    print('for iteration ', i, 'chisq is: ', chi_cur)

data_mcmc = np.empty([len(nstep),2])
data_mcmc[:,0] = chain
data_mcmc[:,1] = chisqs

np.savetxt('chain_chi', data_mcmc)
print('percent accepted: ', npass/nstep*100)
print('final params: ', chain[-1, :])
#print('final chisq: '

