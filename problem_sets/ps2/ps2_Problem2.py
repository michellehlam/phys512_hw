import numpy as np
import matplotlib.pyplot as plt

#-----------------------
#       test function
#-----------------------
def lorentz(x):
    return 1.0/(1.0+x**2)


#-----------------------
#       My version 
#-----------------------
def integrate_adaptive(fun, a,b,tol, extra= None):
    # initial call, extra = None, future calls extra holds sub calls
    # integrates between a, b
    
    # Using simpsons
    x = np.linspace(a,b, 5)
    
    if extra is None:
        y = fun(x) # evaluated 5 points
        n_calls = len(y) # start
       
    else:
        y_new = fun(x[1:4]) # evaluate 3 new points
        y = np.concatenate([[extra[0]], y_new, [extra[1]]]) # get all 5 points
        n_calls = len(y_new) # 3 new function calls

    dx = (b-a)/(len(x)-1)
    coarse_step_area = 2*dx*(y[0]+4*y[2]+y[4])/3
    fine_step_area = dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3
    diff = np.abs(coarse_step_area-fine_step_area)

    if diff<tol:
        return fine_step_area, n_calls
    else:
        # cut interval in half, split big step into 2 steps
        xmid = (a+b)/2
        # tol/2 because smaller step = smaller tolerance
        # extra, choose endpoints for left half
        left, n_calls_left=integrate_adaptive(fun,a,xmid, tol/2, extra = np.array([y[0], y[2]]))
        # choose endpoints for right half
        right, n_calls_right = integrate_adaptive(fun, xmid,b,tol/2,extra=np.array([y[2],y[4]])) 
        n_calls = n_calls + n_calls_left + n_calls_right
        return left+right, n_calls   # total integral 

#----------------------------------------------
# Class version + count function calls
#----------------------------------------------
def integrate_adaptive_class(fun,x0,x1,tol):
    x = np.linspace(x0,x1,5)
    y = fun(x) # evaluated 5 points
    n_calls = len(y)
    dx = (x1-x0)/(float(len(x)-1))
    area1=2*dx*(y[0]+4*y[2]+y[4])/3.0 #coarse step
    area2=dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3.0 #finer step
    err=np.abs(area1-area2)
    if err<tol:
        return area2, n_calls
    else:
        xmid=(x0+x1)/2.0
        left, n_calls_left=integrate_adaptive_class(fun,x0,xmid,tol/2.0)
        right, n_calls_right=integrate_adaptive_class(fun,xmid,x1,tol/2.0)
        n_calls = n_calls + n_calls_left + n_calls_right
        return left+right, n_calls

#-------------------------------------------------------------------------------------------------
#                   Integrate our function
# ------------------------------------------------------------------------------------------------
# choosing our endpoints
xmin = -100
xmax = 100
tolerance = 1e-7

# My integrator
output, fun_evals = integrate_adaptive(lorentz, xmin, xmax, tolerance)
print('Number of function evaluations: ', fun_evals)

# integrator from class
ans, n_evals = integrate_adaptive_class(lorentz, xmin, xmax, tolerance)
print(ans - (np.arctan(xmax)-np.arctan(xmin)))
print('Number of function evaluations (CLASS VERSION) : ', n_evals)
