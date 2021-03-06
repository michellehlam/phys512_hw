import numpy as np
import matplotlib.pyplot as plt

# make correlation function of 2 arrays
# h(x) = \int f(x)*g(x+y)
# correlation function of a gaussian with itself 
# ift(ft(f) conj(ft(g))
def gauss(x):
    sigma =1
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-0.5*x**2/sigma**2)  
    
def correlation(arr1, arr2): # basically autocorrelation , f(x) = g(x)
    f = arr1
    g = arr2
    # normalize
    f=f/f.sum()
    g=g/g.sum()
    
    # have to fft shift, perform correlation in k-space
    h = np.fft.fftshift(np.fft.irfft(np.fft.rfft(f)*np.conjugate(np.fft.rfft(g))))

    plt.plot(f, label = 'original')
    #plt.plot(g, label = 'g')
    plt.plot(h, label = 'correlation function')
    plt.legend()
    plt.show()

    return h

x = np.linspace(-10,10,101)

correlation(gauss(x),gauss(x))