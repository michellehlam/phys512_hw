import numpy as np

# refer to pdf file on explanations of how these terms were arrived at

def num_deriv_4pt(a):
    eps = 2**(-52)              # machine accuracy
    dx = (eps/a**5)**(1/5)      # optimal step size

    x = 42                      # arbitrary test number

    f1 = np.exp(a*(x+dx))       
    f2 = np.exp(a*(x-dx))

    f3 = np.exp(a*(x+2*dx))
    f4 = np.exp(a*(x-2*dx))

    deriv = (8*(f1-f2)-(f3-f4))/(12*dx)     # derivative using the 4 points

    f0 = a*np.exp(a*x)          # true function    
    error = deriv/f0 - 1        # error
    return deriv, error

deriv1,error1 = num_deriv_4pt(1)
deriv2,error2 = num_deriv_4pt(0.01)

print('derivative of f(x) = exp(x) is:', deriv1, 'with error', error1)
print('derivative of f(x) = exp(0.01x) is:', deriv2, 'with error', error2)
