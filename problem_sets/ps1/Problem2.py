import numpy as np

# ndiff:
# input: fun = arbitrary function to differentiate
#        x = evaluate derivative at some x value
#        full == True, ndiff returns derivative, stepsize, and error
#        full == False, ndiff only returns derivative

def ndiff(fun, x,full=False):
    eps = 2**-52                # machine accuracy
    dx = x*eps**(1/3)           # stepsize
    f1 = fun(x+dx)
    f2 = fun(x-dx)
    deriv = (f1-f2)/(2*dx)      # 2-sided derivative
    error = eps**(2/3)*fun(x)   # approximate error
    if full ==True:
        return deriv, dx, error
    else:
        return deriv

# test function
def test(x):
    return np.cos(x)

# result if you choose Full = True
der, step, err = ndiff(test,42,True)
print('If Full = True, we get derivative:',der, ', with a stepsize:',step, ' and our error:' ,err)

# result if you choose Full = False
deriv = ndiff(test,42,False)
print('If Full = False, we get derivative:',deriv) 

