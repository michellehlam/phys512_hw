import numpy as np
import camb
import matplotlib.pyplot as plt

# Data columns:
# 1: multipole
# 2: variance of sky at multipole
# 3: 1 sigma lower uncertainty 
# 4: 1 sigma upper uncertainty

# Assumptions:
# - assume errors are gaussian + uncorrelated
# assume error at each point is average of upper and lower errors


def get_spectrum(pars, lmax = 3000): # taken from planck_likelihood.pyre ',pars)
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
      
# load data
data = np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt')

ell = data[:,0]
spec = data[:,1]
# assume avg error = (upper + lower uncertainty)/2
avg_error = (data[:,2] + data[:,3])/2

# params = [H0, sigma_b*h^2, sigma_c*h^2, tau, As]
params = np.asarray([60,0.02,0.1,0.05,2.00e-9,1.0])

model = get_spectrum(params) 
model = model[:len(spec)]
residuals = spec-model
chisq = np.sum((residuals/avg_error)**2)
print("Original params: chisq is ",chisq," for ",len(residuals)-len(params)," degrees of freedom.")

params = np.asarray([69,0.022, 0.12, 0.06,2.1e-9,0.95])


model = get_spectrum(params)
model = model[:len(spec)]
residuals = spec-model
chisq = np.sum((residuals/avg_error)**2)
print("New params: chisq is ",chisq," for ",len(residuals)-len(params)," degrees of freedom.")


#read in a binned version of the Planck PS for plotting purposes
planck_binned=np.loadtxt('COM_PowerSpect_CMB-TT-binned_R3.01.txt',skiprows=1)
errs_binned=0.5*(planck_binned[:,2]+planck_binned[:,3]);
plt.clf()
plt.plot(ell,model)
plt.errorbar(planck_binned[:,0],planck_binned[:,1],errs_binned,fmt='.')
plt.show()

