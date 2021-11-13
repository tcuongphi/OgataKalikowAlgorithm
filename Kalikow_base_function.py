import numpy as np
from scipy.stats import expon
from scipy.stats import uniform
from scipy.stats import poisson
import matplotlib.pyplot as plt

# FUNCTIONS #######################################################################################################################################################################

# Polt
def EmpiricalDensity(v):
    fig, ax = plt.subplots(1, 1)
    
    x = np.linspace(uniform.ppf(0.01),uniform.ppf(0.99), 100)
    ax.plot(x, uniform.pdf(x),'r-', lw=5, alpha=0.6, label='uniform pdf')

    ax.hist(v, density=True, histtype='stepfilled', alpha=0.2)
    ax.legend(loc='best', frameon=False)
    plt.show()

# Exponential laws
def RandomExp(lam, n):
    return(expon.rvs(scale = 1/lam, size = n))

# Uniform laws
def RandomUnif(left,right,n):
    return(uniform.rvs(loc = left, scale = right-left, size = n))

# Homogeneous Poisson process
def PoissonProcess(start,end,lam):
    out = np.empty((1,0))
    N = poisson.rvs(lam*(end-start), size = 1)
    # out = RandomUnif(start, end, N)
    # out = sorted(out)
    out = RandomUnif(start,end,N)
    ind = out.argsort()
    out = out[ind]
    return(out)

# Homogeneous Poisson process with hard refractory period
def RefractoryPoissonProcess(start,end,lam,delta):
    out = np.empty((1,0))
    t = start + RandomExp(lam, 1)
    while t < end:
        out = np.c_[out,t]
        t = t + delta + RandomExp(lam, 1)
    return(out)

# Discrete random variable with a custom explicit law
def Sample(x,p,size):
    out = np.zeros((size))
    u = RandomUnif(0,1,size)
    for i in range(size):
        w = 0
        cdf = p[0]
        while (u[i] > cdf):
            w = w + 1
            cdf = cdf + p[w]
        out[i] = x[w]
    return(out)

# Same with interger values
def IntSample(x,p,size):
    out = np.zeros((size), int)
    u = RandomUnif(0,1,size)
    for i in range(size):
        w = 0
        cdf = p[0]
        while (u[i] > cdf):
            w = w + 1
            cdf = cdf + p[w]
        out[i] = x[w]
    return(out)

def Point_in_Interval(x, t, a):
    size = len(x)
    out = np.zeros(size)
    out = np.where((x < t) & (x >= t - a))[0]
    return(out)

#######################################################################################################################################################################