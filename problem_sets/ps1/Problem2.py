import numpy as np

def ndiff(fun, x,full=False):
    eps = 2**-52
    dx = x*eps**(1/3)
    f1 = fun(x+dx)
    f2 = fun(x-dx)
    deriv = (f1-f2)/(2*dx)
    error = eps**(2/3)*fun(x)
    if full ==True:
        return deriv, dx, error
    else:
        return deriv

# test 
def test(x):
    return np.exp(0.01*x)
der, step, err = ndiff(test,42,True)
print('derivative:',der, 'w stepsize',step, ' and error' ,err)
