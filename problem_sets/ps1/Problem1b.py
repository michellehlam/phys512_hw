import numpy as np

def num_deriv_4pt(a):
    eps = 2**(-52)
    dx = (eps/a**5)**(1/5)

    x = 42 # arbitrary number

    f1 = np.exp(a*(x+dx))
    f2 = np.exp(a*(x-dx))

    f3 = np.exp(a*(x+2*dx))
    f4 = np.exp(a*(x-2*dx))

    deriv = (8*(f1-f2)-(f3-f4))/(12*dx)

    f0 = a*np.exp(a*x)
    error = deriv/f0 - 1
    return deriv, error

deriv1,error1 = num_deriv_4pt(1)
deriv2,error2 = num_deriv_4pt(0.01)

print('derivative of f(x) = exp(x) is:', deriv1, 'with error', error1)
print('derivative of f(x) = exp(0.01x) is:', deriv2, 'with error', error2)
