import numpy as np
import camb
import matplotlib.pyplot as plt
import time

# Data columns:
# 1: multipole
# 2: variance of sky at multipole
# 3: 1 sigma lower uncertainty
# 4: 1 sigma upper uncertainty

# Assumptions:
# - assume errors are gaussian + uncorrelated
# assume error at each point is average of upper and lower errors


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

def get_derivs(pars, lmax=2507):

    # delta between steps, 1/100th of parameter itself
    delta = pars/100
    # get y-values (original)
    cmb = get_spectrum(pars,lmax)
#    H0=pars[0]
#    ombh2=pars[1]
#    omch2=pars[2]
#    tau=pars[3]
#    As=pars[4]
#    ns=pars[5]
    
    # adding delta to do double-sided derivative    
    cmb_p = np.zeros([len(cmb), len(pars)])
    cmb_p_neg = np.zeros([len(cmb),len(pars)])
    # double sided derivative for each param
    for i in range(len(pars)):
        pars_curr = pars
        pars_curr[i] = pars[i] + delta[i]
        cmb_p[:,i] = get_spectrum(pars_curr, lmax)
        pars_curr[i] = pars[i] - delta[i]
        cmb_p_neg[:,i] = get_spectrum(pars_curr,lmax)
    
    derivs = (cmb_p - cmb_p_neg)/(2*delta)
    return cmb, derivs

# update lambda for levenberg-marquardt
def update_lamda(lamb, success):
    if success: 
        lamb = lamb/1.5
        if lamb<0.5:
            lamb = 0
    else:
        if lamb ==0:
            lamb = 1
        else:
            lamb= lamb*1.5**2
    return lamb

# initialize
m0 = np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95])
data = np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell = data[:,0]
spec = data[:,1]

err =  0.5*(data[:,2]+data[:,3]) # sigma
#inverse noise matrix
Ninv = np.linalg.inv(np.diag(err**2))

# initial chi^2
y_guess = get_spectrum(m0)
y_guess = y_guess[:len(spec)]
r = spec-y_guess
chisq0 = np.sum((r/err)**2)
print('initial chi^2:', chisq0)

# taken from lm code in class
def get_matrices(m,fun,x,y,Ninv=None):
    model,derivs=get_derivs(m)
    model = model[:len(y)]
    derivs = derivs[:len(y),:]

    r=y-model
    if Ninv is None:
        lhs=derivs.T@derivs
        rhs=derivs.T@r
        chisq=np.sum(r**2)
    else:
        lhs=derivs.T@Ninv@derivs 
        rhs=derivs.T@(Ninv@r) # A^T Ninv r
        chisq=r.T@(Ninv@r)    
    return chisq,lhs,rhs

def linv(mat,lamda):
    mat=mat+lamda*np.diag(np.diag(mat))
    return np.linalg.inv(mat)

def fit_lm_clean(m,fun,x,y,Ninv=None,niter=10,chitol=0.01):
#levenberg-marquardt fitter that doesn't wastefully call extra
#function evaluations, plus supports noise
    lamda=0
    chisq,lhs,rhs=get_matrices(m,fun,x,y,Ninv)
    for i in range(niter):
        lhs_inv=linv(lhs,lamda)
        dm=lhs_inv@rhs
        chisq_new,lhs_new,rhs_new=get_matrices(m+dm,fun,x,y,Ninv)
        
        errs = np.sqrt(np.diag(lhs_inv))
        if chisq_new<chisq:  
            #accept the step
            #check if we think we are converged - for this, check if 
            #lamda is zero (i.e. the steps are sensible), and the change in 
            #chi^2 is very small - that means we must be very close to the
            #current minimum
            if lamda==0 and m[3]>0.01:
                if (np.abs(chisq-chisq_new)<chitol):
                    print(np.abs(chisq-chisq_new))
                    print('Converged after ',i,' iterations of LM')
                    return m+dm
            chisq=chisq_new
            lhs=lhs_new
            rhs=rhs_new
            m=m+dm
            lamda=update_lamda(lamda,True)
            
        else:
            lamda=update_lamda(lamda,False)
        print('on iteration ',i,' chisq is ',chisq,' with step ',dm,' and lamda ',lamda)
    return m, errs, linv(lhs,lamda)

#-------------------------------------------------------------------------------
#                               Run program
#-------------------------------------------------------------------------------

start_time = time.time()

data = np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell = data[:,0]
spec = data[:,1]

#m0 = np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95])
m_fit,fit_error, inv_curvature = fit_lm_clean(m0, get_spectrum, ell,spec, Ninv, niter = 10)

param_name = ['H0: ', 'ombh2: ', 'omch2: ','tau: ', 'As: ','ns: ']


for i in range(len(m_fit)):
    print(param_name[i],m_fit[i], ' +/- ', fit_error[i])

param_file = np.empty([len(m_fit), 2])
param_file[:,0] = m_fit
param_file[:,1] = fit_error
np.savetxt('planck_fit_params.txt', param_file)
np.savetxt('param_covar.txt', inv_curvature)

y_pred=get_spectrum(m_fit)
y_pred=y_pred[:len(spec)]
resid=spec-y_pred
chisq=np.sum( (resid/err)**2)
print(chisq)

print('Code took this long: ', (time.time()-start_time)/60, ' minutes') 
