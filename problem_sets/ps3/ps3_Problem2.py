import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# decay chain 
# can use ODE solver from scipy 
# include all dcay products in chain 
# start from pure U238 

# ode solver,
#-------------------------------------------------------------------------------
#                            Part A
#-------------------------------------------------------------------------------


def fun(x,y):
    lambdas = np.array([4.468*1e9*31536000,
    24.1*2486400, 6.7*3600, 245500*31536000, 75380*31536000,
    1600*31536000, 3.8235*2486400,3.1*60, 26.8*60,
    19.9*60, 164.3e-6, 22.3*31536000, 5.015*31536000,
    138.376*2486400]) # seconds

    # lambda is an array of all the decay constants in order
    dydx = np.zeros(15)#len(lambdas)+2) # because last element is stable
    dydx[0] = -y[0]*np.log(2)/lambdas[0]
    for i in range(1, 13):#len(lambdas)):
            dydx[i] = y[i-1]*np.log(2)/lambdas[i-1] - y[i]*np.log(2)/lambdas[i]
    # last decay 
    dydx[14] = y[13]*np.log(2)/lambdas[13]
    #dydx[len(dydx)-1] = y[len(dydx)-2]*lambdas[len(dydx)-2]
    return dydx

lambdas = np.array([4.468*1e9*31536000,
    24.1*2486400, 6.7*3600, 245500*31536000, 75380*31536000,
    1600*31536000, 3.8235*2486400,3.1*60, 26.8*60,
    19.9*60, 164.3e-6, 22.3*31536000, 5.015*31536000,
    138.376*2486400]) # in seconds

y0 = np.zeros(15)#len(lambdas)+2) 
y0[0] = 1
x0 = 0
x1 = lambdas[0]*1
print(y0)
# solve using implicit/stiff method
ans_stiff = integrate.solve_ivp(fun, [x0,x1], y0, method = 'Radau')


#plt.plot(ans_stiff.t/31536000,ans_stiff.y[0,:], '*-', label = 'Uranium-238')
#plt.plot(ans_stiff.t/31536000,ans_stiff.y[14,:], '*-', label = 'Plomb-206')
plt.plot(ans_stiff.t/31536000,ans_stiff.y[5,:], '*-', label = 'Radium-226')
plt.legend()
plt.xlabel('years')
plt.show()

#---------------------------------------------------------------------------------------
#                                Part B
#---------------------------------------------------------------------------------------

print(np.shape(ans_stiff.y))


print(np.shape(ans_stiff.t))

Pb206 = ans_stiff.y[14,0:21]
print(np.shape(Pb206))
U234 = ans_stiff.y[3,0:21]
Th230 = ans_stiff.y[4,0:21]
timerange = ans_stiff.t[0:21]
print(np.shape(timerange))
plt.plot(timerange/31536000, Pb206/U234, '*-')
plt.show()

plt.plot(timerange/31536000, Th230/U234, '*-')
plt.show()

print('last value (Th230/U234):' , ans_stiff.y[4,len(U234)]/ans_stiff.y[3,len(U234)])
print('last value (U234/Th230):' , ans_stiff.y[3,len(U234)]/ans_stiff.y[4,len(U234)])


