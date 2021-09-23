import numpy as np
import matplotlib.pyplot as plt

def integrate_adaptive(fun, a,b,tol, extra= None):
    # initial call, extra = None, future calls extra holds sub calls
    # integrates between a, b
    
    # Using simpsons
    x = np.linspace(a,b, 5)
    if extra is None:
        y = fun(x) # evaluated 5 points
    else:
        y_new = fun(x[1:4]) # evaluate 3 new points
        y = np.concatenate([[extra[0]], y_new, [extra[1]]]) # get all 5 points
        
    dx = (b-a)/(len(x)-1)
    coarse_step_area = 2*dx*(y[0]+4*y[2]+y[4])/3
    fine_step_area = dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3
    diff = np.abs(coarse_step_area-fine_step_area)

    if diff<tol:
        return fine_step_area
    else:
        # cut interval in half, split big step into 2 steps
        xmid = (a+b)/2
        # tol/2 because smaller step = smaller tolerance
        # extra, choose endpoints for left half
        left=integrate_adaptive(fun,a,xmid, tol/2, extra = np.array([y[0], y[2]]))
        # choose endpoints for right half
        right = integrate_adaptive(fun, xmid,b,tol/2,extra=np.array([y[2],y[4]])) 
        return left+right # total integral 

def lorentz(x):
    return 1/(1+x**2)

xmin = -100
xmax = 100
output = integrate_adaptive(lorentz, xmin, xmax, 1e-7)
print(output-(np.arctan(xmax)-np.arctan(xmin)))
