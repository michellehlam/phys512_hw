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

# fourier transform of window function
ft_win = np.abs(np.fft.fft(win))

print('first point: ', ft_win[0], 'and second point: ', ft_win[1])
print('and last point: ', ft_win[-1])

