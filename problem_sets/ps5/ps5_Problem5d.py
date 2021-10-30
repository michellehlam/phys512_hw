import numpy as np 
import matplotlib.pyplot as plt

# goal: get rid of spectral leakage using window function 
# 0.5 - 0.5cos(2pi x/N)

# set up non-integer fft
N= 1024
x = np.arange(N)
k = 15.4
y = np.cos(2*np.pi*x*k/N)
yft = np.fft.fft(y)

# use window function
xx = np.linspace(0,1,N)*2*np.pi
win = 0.5 - 0.5*np.cos(xx)
yft_win = np.abs(np.fft.fft(win*y))

# plot windowing vs no windowing
plt.plot(np.abs(yft), label = 'no window')
plt.plot(yft_win, label = 'with window')
plt.legend()
plt.show()

# zoomed in plot
#plt.plot(np.abs(yft), label = 'no window')
#plt.plot(yft_win, label = 'with window')
#plt.xlim(0,50)
#plt.legend()
#plt.show()

