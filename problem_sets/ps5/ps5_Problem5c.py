import numpy as np 
import matplotlib.pyplot as plt

# write DFT of non-integer sine wave
# use non-integer k
def fft_anal_sin(k1, N):
    k0 = np.arange(N)
    left = (1- np.exp(-2*np.pi*1j*(k0-k1)))/(1-np.exp(-2*np.pi*1j*(k0-k1)/N))
    right = (1- np.exp(-2*np.pi*1j*(k0+k1)))/(1-np.exp(-2*np.pi*1j*(k0+k1)/N))
    return ((left-right)/2/1j)

    # sin(x) = (e^{ix} - e^{-ix})/(2i)
    # k-k' in exponent

# normally FFT of sin = delta
k1 = 20.25#70.6
N = 1000#500
x = np.arange(N)

# take dft using numpy
fun = np.sin(2*np.pi*k1*x/N)
fft_sin = (np.fft.fft(fun))
# take our analytical fft
test = fft_anal_sin(k1,N)

# compare numpy fft vs analytical
print('average of residuals: ', np.mean(np.abs((test-fft_sin))))

# plot both ffts
plt.plot(np.abs(fft_sin), label = 'numpy fft')#/np.std(fft_sin)**2)
plt.plot(np.abs(test), label = 'analytical')#/np.std(test)**2)
plt.legend()
#plt.xlim(60,120)
plt.show()

# plot residuals
plt.plot(np.abs(fft_sin-test), label = 'residuals')#/np.std(test)**2)
plt.legend()
#plt.xlim(60,120)
plt.show()