import numpy as np
import matplotlib.pyplot as plt
import random

# couldn't load Jon's code/ctypes library, so used Jon's output
file = np.loadtxt('rand_points.txt')
x = file[:,0]
y = file[:,1]
z = file[:,2]

# 2D plotting
a = 1
b = 1
plt.plot(a*x + b*y,z, '.')
plt.xlabel('x+y')
plt.ylabel('z')
plt.title('ctypes')
plt.show()

# comparing with python's random number generator
n = len(z)
vec = np.zeros([n*3,1])
for i in range(n*3):
    vec[i] = random.randint(0,int(1e8)+1)
vec = np.reshape(vec, [n,3])
x = vec[:,0]#np.random.rand(n)*1e8
y = vec[:,1]#np.random.rand(n)*1e8
z = vec[:,2]#np.random.rand(n)*1e8

# plot python random numbers
plt.plot(a*x + b*y,z, '.')
plt.title('python')
plt.xlabel('x+y')
plt.ylabel('z')
plt.show()
