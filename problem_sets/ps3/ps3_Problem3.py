import numpy as np
import matplotlib.pyplot as plt

file = np.loadtxt('dish_zenith.txt') # contains x,y,z positions in mm

# Part A in pdf

# ------------------------------------------------------------------
#                        Part  B
# ------------------------------------------------------------------

# Description: we have rotationally symmetric paraboloid: z - z0 = a ((x-x0)^2 - (y-y0)^2)
#              Solve for x0, y0, z0, a
#            - Using linear least squares, we redefine the parameters
# <d> = Am 
# chi^2 = (d-Am)^T N^-1 (d-Am)
# m = (A^T N^-1 A)^-1 A^T N^-1 d 
# assume N = I in part B

x = file[:,0]
y = file[:,1]
z = file[:,2]

# setting up our matrix A
A = np.zeros([len(x),4])

A[:,0] = x**2 + y**2
A[:,1] = x
A[:,2] = y
A[:,3] = np.ones(len(y))

A_transp = np.matrix.transpose(A)
m = np.linalg.inv(A_transp@A)@A_transp@z

# Redefine parameters we want
a = m[0]
x0=-m[1]/2/a
y0 = -m[2]/2/a
z0 = m[3] - (x0**2*a + y0**2*a + a)

print('a:', a)
print('x0:', x0)
print('y0:',y0)
print('z0:',z0)

# lets calculate z with our fit parameters
z_pred = A@m

#plt.plot(z_pred, z, '.')
#plt.show()

# plot residuals
plt.plot(z-z_pred)
plt.ylabel('residuals')
plt.show()
# --------------------------------------------------------------
#                        Part C
#---------------------------------------------------------------
print('noise in data:',np.std(z-z_pred)) # estimate of noise (not error) in data

# estimate noise in parameter fits
# calculate error in a
noise = np.std(z-z_pred)  # estimate noise from data 
N = np.eye(len(z))*noise**2
Ninv = np.eye(len(x))*noise**-2

# parameter covariance thing to get uncertainty/error bars
err_m = np.linalg.inv(A_transp@Ninv@A)
err_m_noN = np.linalg.inv(A_transp@A)

errs = np.sqrt(np.diag(err_m))
errs_noN = np.sqrt(np.diag(err_m_noN))
#print('Error in fit parameters:', errs)

print('Error if estimate N:')
print('a:', a, ' +/- ', errs[0])
print('x0:', x0,' +/- ',errs[1])
print('y0:',y0, ' +/- ', errs[2])
print('z0:',z0, ' +/- ', errs[3])
print('')
print('Error if N=1:')
print('a:', a, ' +/- ', errs_noN[0])
print('x0:', x0,' +/- ',errs_noN[1])
print('y0:',y0, ' +/- ', errs_noN[2])
print('z0:',z0, ' +/- ', errs_noN[3])
print('')


# focal length = 1/4a
f = 1/4/a
df = errs[0]/a*f
df_noN = errs_noN[0]/a*f

print('Calculated from estimate noise:')
print('focal length: ', f, '+/-' , df)
print('')
print('Calculated with N=1:')
print('focal length: ', f, '+/-' , df_noN)


