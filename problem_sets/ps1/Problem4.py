import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

# Comparison of different fits:
# 1. polynomial interpolation performed by function poly_interp (similar to class notes)
# 2. spline interpolation will be done by scipy.interpolate.splrev as allowed by Jon apparently
# 3. rational function interpolation performed by function rat_interp (similar to class notes)
#---------------------------------------------------------------------------------

#1. polynomial interpolation
def poly_interp(x,y,x_test):
    npt = len(x)                                # define number of points
    X = np.empty([npt, npt])                    # set up matrix
    for i in range(npt):
        X[:,i] = x**i                           # fill matrix by column using original x points
    inv_matx = np.linalg.inv(X)                 # invert matrix
    c = inv_matx@y                              # get coefficients of y = Xc solution
    X_test = np.empty([len(x_test), npt])       # set up matrix for new x-values for interpolation
    for i in range(npt):                        
        X_test[:,i] = x_test**i                 # fill matrix by column with new x-values
    return X_test@c                             # return new interpolated y-values at new x-values

#2. spline interpolation
def spline_interp(x,y,x_test):
    spline_fit = interp.splrep(x,y)                     # call spline fitting function
    y_test = interp.splev(x_test, spline_fit)           # apply fit on new x-values, x_test, return interpolated y-values
    return y_test

#3. rational interpolation
def rat_interp(x,y,x_test,use_pinv):        
# 1) fit data to determine 'function'/'fitting model'
# 2) evaluate 'fitted model' at new x-value
# Pt. 1
    npt = len(x) # number of points
    m = len(x)//2 # to make it symmetrical
    n = len(x)-m-1
     
    assert(len(y)==npt)
    assert(n+m+1==npt)
    top_mat = np.zeros([npt, n+1])              # "+ 1" to account for x^0
    bottom_mat = np.zeros([npt,m])              # no x^0 term

    for i in range(n+1): 
        top_mat[:,i] = x**i                     # numerator
    for i in range(m):
        bottom_mat[:,i] = -y*x**(i+1)           # denominator, skipping the x^0 term
    matx = np.hstack([top_mat, bottom_mat])
    
    # to get p,q multiply inverse matx with y (temperature values)
    # use_pinv == True, then we deal with singularities
    if use_pinv==True:
        output = np.linalg.pinv(matx)@y
    else:
        output = np.linalg.inv(matx)@y
    p = output[:n+1]                            # the first n+1 elements belong to p, top
    q = output[n+1:]                            # rest of the m elements belong to q, bottom
    #print('p:',p)
    #print('q:',q)
    
    # now that we have p,q we can interpolate our temperature
    top = 0 # starting point
    for i,ith_element in enumerate(p):
        top = top + ith_element*x_test**i               # polynomial, numerator of rational function
    bottom = 1 # starting point
    for i, ith_element in enumerate(q):
        bottom = bottom + ith_element*x_test**(i+1)     # polynomial, denominator of rational function
    return top/bottom                                   # quotient, return interpolated y 



#------------------------------------------------------------------------------------
# Pt 1: cosine function
# our data points:
x = np.linspace(-np.pi/2, np.pi/2, 10)
y = np.cos(x)

test_x = np.linspace(-np.pi/2, np.pi/2, 1001)
pred_y_rat = rat_interp(x,y,test_x,use_pinv=False)
pred_y_poly = poly_interp(x,y,test_x)
pred_y_spline = spline_interp(x,y,test_x)
plt.scatter(x,y)
plt.plot(test_x, pred_y_rat, label = 'rational interp')
plt.plot(test_x, pred_y_poly, label = 'polynomial interp')
plt.plot(test_x, pred_y_spline, label = 'spline')
plt.legend()
plt.title('Cosine function')
plt.show()

#------------------------------------------------------------------------------------
# Pt 2: lorentzian function
# data points:
x = np.linspace(-1,1,8)
y = 1/(1+x**2)

test_x = np.linspace(-1, 1, 1001)
pred_y_rat = rat_interp(x,y,test_x,use_pinv = True)
pred_y_poly = poly_interp(x,y,test_x)
pred_y_spline = spline_interp(x,y,test_x)
plt.scatter(x,y)
plt.plot(test_x, pred_y_rat, label = 'rational interp')
plt.plot(test_x, pred_y_poly, label = 'polynomial interp')
plt.plot(test_x, pred_y_spline, label = 'spline')
#plt.plot(test_x, pred_y_rat, label = 'rational interp')
plt.title('lorentzian function')
plt.legend()
plt.show()

# Pt 2.5: Residuals/error in rational interpolation
plt.plot(test_x, pred_y_rat-1/(1+test_x**2), label ='with pinv')
plt.plot(test_x, rat_interp(x,y,test_x,use_pinv=False)-1/(1+test_x**2), label ='inv')
plt.title('Lorentzian, rational interp residuals/error')
plt.legend()
plt.show()
