import numpy as np 
import matplotlib.pyplot as plt

# write DFT of non-integer sine wave
# use non-integer k
def fft_anal_sin(k0, k1, N):
    left = (1- np.exp(-2*np.pi*1j*(k0-k1)))/(1-np.exp(-2*np.pi*1j*(k0-k1)/N))
    right = (1- np.exp(-2*np.pi*1j*(k0+k1)))/(1-np.exp(-2*np.pi*1j*(k0+k1)/N))
    return np.abs((left-right)/2/1j)

    # sin(x) = (e^{ix} - e^{-ix})/(2i)
    # k-k' in exponent

# normally FFT of sin = delta
k1 = 70.6
N = 500
x = np.linspace(0, N,N)
k0 = np.linspace(0, N,N) #(1-x/k1) #np.linspace(0,100,1000)

fun = np.sin(2*np.pi*k1*x/N)
fft_sin = np.abs(np.fft.fft(fun))

test = fft_anal_sin(k0,k1,N)
plt.plot(fft_sin, label = 'numpy fft')#/np.std(fft_sin)**2)
plt.plot(test, label = 'analytical')#/np.std(test)**2)
plt.legend()
#plt.xlim(60,120)
plt.show()
