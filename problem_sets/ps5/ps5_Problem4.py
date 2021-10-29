import numpy as np 
import matplotlib.pyplot as plt

# convolution of 2 arrays
# w/o wrap arround , can add zeros to end of input 
# f, g not necessarily the same length

# arbitrary gauss function
def gauss(x,sigma):
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-0.5*x**2/sigma**2)  

def conv_safe(f,g):
     # make delta function
    #g=0*f # 0 everywhere else
    #g[int(0.25*len(f))]=1 # except at shift point

    # normalize
    f=f/f.sum()
    g=g/g.sum()
    
    zeros = np.zeros(int(len(f)*0.5))
    f = np.append(f,zeros)
    zeros2 = np.zeros(int(len(f)-len(g)))
    g = np.append(g,zeros2)
    # go to k-space to perform convolution
    h=np.fft.irfft(np.fft.rfft(f)*np.fft.rfft(g),len(f))
    return h

x1 = np.linspace(-10,10,75)
x2 = np.linspace(-5,5,60)
f = gauss(x1,2.5)
plt.plot(f/f.sum(), label = 'f')
g = gauss(x2,0.25)
plt.plot(g/g.sum(), label = 'g')
h = conv_safe(f,g)

plt.plot(h/h.sum(), label = 'h')
print('output length:', len(h))

plt.legend()
plt.show()
