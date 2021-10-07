import numpy as np
import matplotlib.pyplot as plt

def der(x,y): # derivative we want to integrate
    #return -y
    return y/(1+x**2)

def rk4_step(fun,x,y,h):
    k0=fun(x,y)*h
    k1=fun(x+h/2,y+k0/2)*h
    k2=fun(x+h/2,y+k1/2)*h
    k3=fun(x+h,y+k2)*h
    return (k0+2*k1+2*k2+k3)/6


def rk4_stepd(fun,x,y,h):   #dy
    # use multiple steps
    y1 = rk4_step(fun,x,y,h) # = y_true + err
    # take two half steps
    y2a = rk4_step(fun,x,y, h/2) # = y_true + err/16 ....?
    y2b = rk4_step(fun, x+h/2, y + y2a, h/2)
    y2 = y2a + y2b
    return y2+(y2-y1)/15

nstep = 200
nstep_d = int(200/3)

x = np.linspace(-20,20,nstep+1)
x_d = np.linspace(-20,20,nstep_d+1)

y_true = np.exp(np.arctan(x))
y_true_d =np.exp(np.arctan(x_d))
y = 0*x
y[0] =0.2
print(x[0])
print(y[0])
y_d = 0*x_d
y_d[0] = 1

for i in range(nstep):
    h = x[i+1]-x[i]
    y[i+1] = y[i] + rk4_step(der, x[i], y[i],h)

for i in range(nstep_d):
    h = x_d[i+1] - x_d[i]
    y_d[i+1] = y_d[i] + rk4_stepd(der, x_d[i], y_d[i],h)

print('1step:',np.std(y-y_true))
print('more steps:',np.std(y_d-y_true_d))

plt.plot(x,y_true, label = 'true')
plt.plot(x,y, label = '1 step')
plt.plot(x_d,y_d, label = 'more steps')
plt.legend()
plt.show()
