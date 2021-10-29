import numpy as np 
import matplotlib.pyplot as plt 

# correlation + convolution 
# take autocorrelation of shifted gaussian

# get shifted function
def conv(array, shift):
    f = array
    # make delta function
    g=0*f # 0 everywhere else
    g[int(shift)]=1 # except at shift point

    # normalize
    f=f/f.sum()
    g=g/g.sum()

    # go to k-space to perform convolution
    h=np.fft.irfft(np.fft.rfft(f)*np.fft.rfft(g),len(array))
    return h

# get correlation function
def corr(arr1, arr2):
    f = arr1
    g = arr2
    f=f/f.sum()
    g=g/g.sum()

    h = np.fft.fftshift(np.fft.irfft(np.fft.rfft(f)*np.conjugate(np.fft.rfft(g))))
    return h

# arbitrary gauss function
def gauss(x):
    sigma =1
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-0.5*x**2/sigma**2)  
    
x = np.linspace(-10,10,100)
arr = gauss(x)
shift = len(x)*0.5 #choose arbitary shift
shifted_gauss = conv(arr, shift)

autocorr = corr(shifted_gauss, shifted_gauss) # autocorrelation

plt.plot(arr/arr.sum(), label = 'og gauss')
plt.plot(shifted_gauss, label = 'shifted gauss')
plt.plot(autocorr, label = 'autocorrelation')
plt.legend()
plt.show()

