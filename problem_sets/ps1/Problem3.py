import numpy as np 
import scipy.interpolate as interp
import matplotlib.pyplot as plt

print(np.__version__)

# import data
dat = np.loadtxt('lakeshore.txt')

def lakeshore(V, data):
    x = data[:,1]                           # grab 2nd column for voltage 
    y = data[:,0]                           # grab 1st column for temperature 
    x =x[::-1]                              # reverse array, so numbers ascend
    y = y[::-1]                             # reverse array to match 
    spline_fit = interp.splrep(x,y)         # call library function as done in class
    temp_at_V = interp.splev(V, spline_fit) # interpolate temperature from spline fit
    
    # random number generator
    rng = np.random.default_rng(seed=12345)
    N_resamp = 10
    N_samp = 40
    
    intp_y_samp = []
    for i in range(N_resamp):
        ind = list(range(x.size))
        # randomly choose N_samp samples
        to_interp = rng.choice(ind, size=N_samp, replace = False )
        to_interp.sort() # increasing x

        # interpolate again
        resampled_fit = interp.splrep(x[to_interp],y[to_interp])
        resampled_y = interp.splev(V, resampled_fit)
        intp_y_samp.append(resampled_y)

    # Stats on resampling
    intp_y_samp = np.array(intp_y_samp)
    std = np.std(intp_y_samp, axis = 0)
    error2 = np.nanmean(std)
    error2_std = np.nanstd(std)
    print('std: ', (error2), '+/- ', (error2_std))

    return temp_at_V, error2

test_V = np.linspace(0.1, 1.6, 1001)
test_T, error_T = lakeshore(test_V, dat)

plt.plot(dat[:,1], dat[:,0], '+')
plt.plot(test_V, test_T)
plt.show()
