import numpy as np
import matplotlib.pyplot as plt

# convolution f*g = f(t)*g(t-tau) 
# f = array, g = delta function
# function "conv" takes array and amount to shift array
# move to k-space so you can just multiply the 
def conv(array, shift):

    f = array
    x = np.linspace(0, len(f), len(f))

    # make delta function
    g=0*x # 0 everywhere else
    g[int(shift)]=1 # except at shift point

    # normalize
    f=f/f.sum()
    g=g/g.sum()

    # go to k-space to perform convolution
    h=np.fft.irfft(np.fft.rfft(f)*np.fft.rfft(g),len(array))

    # plot graph
    plt.plot(f,'r', label = 'f(x)')
    plt.plot(h, 'k', label = 'h(x)')
    plt.legend()
    plt.show()
    return h

# gaussian function for our array
def gauss(x):
    sigma = 1
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-0.5*x**2/sigma**2)

x = np.linspace(-10,10, 100)
arr = gauss(x)
shift = len(x)*0.25
h = conv(arr,shift)