import numpy as np
import matplotlib.pyplot as plt
# Define x,y data points
npt = 2000
x= np.linspace(0.5,1, npt)
y = np.log2(x)
deg =6 

global ch_fit#, leg_fit
x_rescale = np.linspace(-1,1,npt) # stretches out our function, since cheb is defined from -1 to 1
ch_fit = np.polynomial.chebyshev.chebfit(x,y,deg)
#leg_fit = np.polynomial.legfit(x_rescale,y,deg)


def mylog2(x): # now we evaluate from 0.5 to     

# note: log_a(x) = ln(x)/ln(a)

    # break into mantissa + exponent
    mantissa, exp = np.frexp(x)
    mantissa_e, exp_e = np.frexp(np.e)
    # gives us log2(x)
    y_pred_ch = np.polynomial.chebyshev.chebval(mantissa,ch_fit)
    y_pred_ch_e = np.polynomial.chebyshev.chebval(mantissa_e, ch_fit)
# ln(x) = log_2(x)(ln(2))
# ln(2) = 1/log_2(x)

    return (y_pred_ch+ exp)*1/(y_pred_ch_e+exp_e)
 #   y_pref_leg = np.polynomial.legval(x,leg_fit)
#x = np.linspace(0,100,100)
x = np.linspace(0.1,10,100)
error = np.abs(np.log(x)-mylog2(x))
print(error.mean())

plt.scatter(x,mylog2(x))
plt.plot(x,np.log(x))
plt.show()
