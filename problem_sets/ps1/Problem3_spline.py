import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

dat = np.loadtxt('lakeshore.txt')
volt = dat[:,1] # volt
temp = dat[:,0] # temp
x = volt[::-1]
y = temp[::-1]

npt = len(x)
X=np.zeros([npt,4])
for i in range(4):
    X[:,i] = x**i
X_T = X.transpose()
Xinv = np.linalg.inv(X_T@X)
#Xinv = np.linalg.inv(X)
c = Xinv@X_T@y

x_test = np.linspace(x[1], x[-1],npt*100)
X_test = np.zeros([len(x_test),4])
for i in range(4):
    X_test[:,i]=x_test**i
y_test = X_test@c
plt.plot(x,y, '+')
plt.plot(x_test, y_test)
plt.show()


