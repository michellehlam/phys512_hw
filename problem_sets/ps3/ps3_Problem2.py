import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Description:
# -decay chain , can use ODE solver from scipy 
# -include all decay products in chain 
# -start from pure U238, ie. 100% U238 initially 
# -Using implicit or stiff integration since the decay rates are wildly different


#-------------------------------------------------------------------------------
#                            Part A
#-------------------------------------------------------------------------------
# convert all decay rates to time
year_to_s = 31536000
day_to_s = 2486400
hour_to_s = 3600

# define decay constants of our decay chain
lambdas = np.log(2)/np.array([4.468e9*year_to_s,
    24.1*day_to_s, 6.7*hour_to_s, 245500*year_to_s, 75380*year_to_s,
    1600*year_to_s, 3.8235*day_to_s,3.1*60, 26.8*60,
    19.9*60, 164.3e-6, 22.3*year_to_s, 5.015*year_to_s,
    138.376*day_to_s]) # seconds

# differential equations of our chain
def fun(x,y):
    # lambda is an array of all the decay constants in order
    dydx = np.zeros(15) # because last element is stable so not incl. in lambdas list
    
    # first decay
    dydx[0] = -y[0]*lambdas[0]
    
    # middle decay chain
    for i in range(1, 14):#len(lambdas)):
            dydx[i] = y[i-1]*lambdas[i-1] - y[i]*lambdas[i]
    
    # last decay, stable
    dydx[len(dydx)-1] = y[len(lambdas)-1]*lambdas[len(lambdas)-1] 
    return dydx


# initialize
y0 = np.zeros(len(lambdas)+1) 
y0[0] = 1
x0 = 0
x1 = np.log(2)/lambdas[0]*2 # arbitrarily chose two half lives

# solve using implicit/stiff method
ans_stiff = integrate.solve_ivp(fun, [x0,x1], y0, method = 'Radau')

# Plot to check it makes sense
plt.plot(ans_stiff.t/year_to_s,ans_stiff.y[0,:], '*-', label = 'Uranium-238')
#plt.plot(ans_stiff.t/year_to_s,ans_stiff.y[14,:], '*-', label = 'Plomb-206')
#plt.plot(ans_stiff.t/year_to_s,ans_stiff.y[5,:], '*-', label = 'Radium-226')
plt.legend()
plt.xlabel('years')
plt.show()

#---------------------------------------------------------------------------------------
#                                Part B
#---------------------------------------------------------------------------------------

# Grab decays of interest to calculat ratios
Pb206 = ans_stiff.y[14,:]
U238 = ans_stiff.y[0, :]
U234 = ans_stiff.y[3,0:21]
Th230 = ans_stiff.y[4,0:21]

# plot interesting parts of the graph
timerange = ans_stiff.t[:]
timerange_short = ans_stiff.t[0:21]

# analytical solution for Pb206/U238
Pb206_U238_ratio = 1/(np.exp(-lambdas[0]*timerange))-1

# Plot stiff and analytical solution for ratio
plt.plot(timerange/year_to_s, Pb206/U238, '*-', label = 'stiff')
plt.plot(timerange/year_to_s,Pb206_U238_ratio, '+', label = 'analytical')
plt.legend()
plt.ylabel('Pb206/U238')
plt.xlabel('years')
plt.show()

# Plot Th340/U234 graph
plt.plot(timerange_short/year_to_s, Th230/U234, '*-')
plt.ylabel('Th230/U234')
plt.xlabel('years')
plt.show()

# Check long term ratio of Th340/U234
print('last value (U234/Th230):' , ans_stiff.y[3,len(ans_stiff.t)-1]/ans_stiff.y[4,len(ans_stiff.t)-1])



