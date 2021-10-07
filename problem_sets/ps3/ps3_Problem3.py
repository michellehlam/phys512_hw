import numpy as np
import matplotlib.pyplot as plt

file = np.loadtxt('dish_zenith.txt') # contains x,y,z positions in mm

# ------------------------------------------------------------------
#                        Part A 
# ------------------------------------------------------------------

# Description: we have rotationally symmetric paraboloid: z - z0 = a ((x-x0)^2 - (y-y0)^2)
#              Solve for x0, y0, z0, a
#            - Pick new set of parameters to make the problem linear,  what are they?

# <d> = Am 
# chi^2 = (d-Am)^T N^-1 (d-Am)
# m = (A^T N^-1 A)^-1 A^T N^-1 d 

x = file[:,0]
y = file[:,1]
z = file[:,2]

print(np.shape(x))
A = np.zeros([len(x),4])

A[:,0] = x**2 + y**2
A[:,1] = x
A[:,2] = y
A[:,3] = np.ones(len(y))

print(np.shape(A))


A_transp = np.matrix.transpose(A)
m = np.linalg.inv(A_transp@A)@A_transp@z
print(m)

a = m[0]
x0=m[1]/2/a
y0 = m[2]/2/a
z0 = m[3] - (x0**2*a + y0**2*a + a)

print('a:', a)
print('x0:', z0)
print('y0:',y0)
print('z0:',z0)


#def mat(x,y,z, params):
#    A[i,:] = 

#plt.plot(x)
#plt.show()

#1, ix,y,z - A123 m = a, z0, y0,z0
