import numpy as np
import matplotlib.pyplot as plt

# Description: This script models log base 2 of x from 0.5 to 1 with a chebyshev fit, and using the fit, one can call the function 'mylog' to take the natural log of any positive number
#-------------------------------------------------------------------------


# Define x,y data points and order
npt = 1000
x= np.linspace(0.5,1, npt)
y = np.log2(x)
deg =7 

global ch_fit
# cheb is defined from -1 to 1, so generally we need to account for rescaling 
# but the built-in function accounts for re-scaling, so we can skip that
# calling numpy's chebyshev built-in fitting function
ch_fit = np.polynomial.chebyshev.chebfit(x,y,deg)

def mylog(x): # takes the natural log of x

# to get natural log from log2, need to do some algebra
# note: log_a(x) = ln(x)/ln(a)
#       ln(z) = ln_2(x)*ln2()
#       ln(2) = 1/log_2(x)
    # break into mantissa + exponent
    mantissa, exp = np.frexp(x)
    mantissa_e, exp_e = np.frexp(np.e)
    # out fit gives us log_2(x)
    y_pred_ch = np.polynomial.chebyshev.chebval(mantissa,ch_fit)
    y_pred_ch_e = np.polynomial.chebyshev.chebval(mantissa_e, ch_fit)

    return (y_pred_ch+ exp)*1/(y_pred_ch_e+exp_e) # ln(z) = ln_2(x)*1/log_2(x)

# test fit
x_test = np.linspace(0.1, 100,200)
y_test = mylog(x_test)
error = np.abs(np.log(x_test)-mylog(x_test)) # error from test fit
print(error.mean()) # print mean error

# plotting
plt.plot(x_test,y_test, label = 'chebyshev')
plt.plot(x_test,np.log(x_test), '+',label ='true')
plt.title('Natural log')
plt.legend()
plt.show()
