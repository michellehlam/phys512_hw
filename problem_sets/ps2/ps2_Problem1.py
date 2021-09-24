import scipy.integrate as integ
import numpy as np
import matplotlib.pyplot as plt
# compare your integrator with scipy.integrate.quad

# Get legendre coefficients
def legendre_coeff(n):
    # Legendre polynomial order = n
    # yi = sum(cj*Pj(xi) --> y = Pc matrix form
    # if P is square + invertible, c = P^-1 y 
    
    x = np.linspace(-1,1, n) # legendre defined from -1,1
    legendre_mat = np.polynomial.legendre.legvander(x,n-1) # get matrix of legendre polynomials
    inv_mat = np.linalg.inv(legendre_mat) # invert matrix
    coeffs = inv_mat[0,:] # pull c0
    return coeffs*(n-1) 
    
#print(legendre_coeff(3))

def integrator(func, xmin, xmax,dx_targ, order):
# using legendre method, no adaptive stepsize, 
    coeffs = legendre_coeff(order+1)

    npt = int((xmax-xmin)/dx_targ)+1 # number of points need to achieve dx_targ
    nn = (npt-1)%(order)
    if nn>0:
        npt = npt+(order-nn)
    
    assert(npt%(order)==1)
    
    x = np.linspace(xmin, xmax, npt)
    dx = x[1] - x[0]
    dat = func(x)

    # reshape data into column 
    mat = np.reshape(dat[:-1], [(npt-1)//order, order]).copy()
    mat[0,0] = mat[0,0] + dat[-1]
    mat[1:,0] = 2*mat[1:,0] # double in first column, since it appears as last element in previous

    vec = np.sum(mat, axis = 0)
    tot = np.sum(vec*coeffs[:-1])*dx
    return tot

# -------------------------------------------------------------------------------------------------
#        Evaluating function
#-----------------------------------------

# plot Ez as function of distance from centre of sphere, want z<R and z>R
# let R = 1 
Q = 1e-9
epsilon0 = 8.85e-12
R =1
sigma = Q/4/np.pi/R**2


# Plot integrator
def Ez_integ(z,r):
    def Ez(u):
        return (z-r*u)/(r**2+z**2-2*r*z*u)**(3/2)
    return r**2*sigma/2/epsilon0*integrator(Ez, -1,1, 1e-4, order=2)
    
# Plot using scipy.integrate.quad
def Ez_quad(z,r):
    def Ez(u):
        return (z-r*u)/(r**2+z**2-2*r*z*u)**(3/2)
    return r**2*sigma/2/epsilon0*integ.quad(Ez,-1, 1)[0]
x = np.linspace(0, R+1, 100)
y_quad = 0*x
y_integ = 0*x

for i in range(len(x)):
    y_quad[i] = Ez_quad(x[i],R)
    y_integ[int(i)] = Ez_integ(x[int(i)],R)

# analytical answer
x_left = np.linspace(0,R,50)
x_right = np.linspace(R, R+1, 50)
true_right = 1/4/np.pi/epsilon0*sigma*4*np.pi*R**2/x_right**2 # from z>R 

# plotting
plt.plot(x,y_quad, label ='quad')
plt.plot(x_left, 0*x_left, 'r')
plt.plot(x_right,true_right, 'r',label = 'true')
plt.plot(x, y_integ, label = 'integrator')
plt.ylabel('Ez')
plt.xlabel('z')
plt.legend()
plt.show()


