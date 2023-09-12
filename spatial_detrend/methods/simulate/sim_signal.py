import numpy as np
import numpy.random as random

#Functions to emulate various astrophysical signals

def rand_walk(len_):
    """
    Generate a random walk time series.
    
    Args:
    len_ : Length of the time series.

    Returns:
    A random walk time series.
    """
    x = w = np.random.normal(size=len_)
    for t in range(len_):
        x[t] = x[t-1] + w[t]
    return x
    
def vec_shock(len_):
    """
    Generate a synthetic time series featuring a "shock" event.
    
    The function models a "shock" event at a random time and overlays it
    with a random walk. The function employs exponential decay pre- and
    post-shock to model the event.

    Args:
    len_ : Length of the time series.

    Returns:
    A shock time series.
    """
    vec_ = np.zeros(len_)
    shock_time = random.randint(1, len_-2)
    after_shock = np.exp(-1*10*np.arange(len_-shock_time)/len_)
    pre_shock = 2*np.exp(-2*20*np.arange(shock_time)[::-1]/len_)
    pre_shock = (pre_shock/np.max(pre_shock))*np.max(after_shock)
    vec_[:shock_time] = pre_shock
    vec_[shock_time:] = after_shock
    bla = rand_walk(len_)
    return -bla*vec_

def soft_transit(len_, period, epoch, dur, alpha = 1):
    """
    Generate a simulated exoplanet transit light curve with a "soft" transit template.
    Not physically simulated (can replace with TLS template or other
    
    Args:
    len_ : Length of the time series.
    period : Period of the transit.
    epoch : Epoch of the transit.
    dur : Duration of the transit.
    alpha : Transit depth. Default is 1.

    Returns:
    Simulated transit light curve with a "soft" ingress and egress.
    """

    transit_lc = np.zeros(len_)
    half_dur = dur/2
    t_times = np.arange(1+((len_-(epoch+dur))//period))*period
    transit_s = -1*np.ones(dur)
    taper_dist = int(half_dur)
    taper = 0.4*np.exp(-.3*np.arange(taper_dist)) - np.ones(taper_dist)
    transit_s[:taper_dist] = taper
    transit_s[dur-taper_dist:] = taper[::-1]
    for t in t_times:
        transit_lc[int(epoch + t - half_dur): int(epoch + t + half_dur)] = transit_s
    return transit_lc*alpha

def box_transit(times_, period, dur, t0, alpha=1):
    """
    Generate a transit signal time-series with box function evaluated at given times.
    
    Args:
    times_ :  Array of time points at which to evaluate the transit time-series
    period : Period of the transit
    dur : Duration of the transit.
    t0 : Epoch.
    alpha : Transit depth. Default is 1.
        
    Returns:
    Transit time series evaluated at `times_`.
    """

    return np.piecewise(times_, [((times_-t0+(dur/2))%period) > dur, ((times_-t0+(dur/2))%period) <= dur], [0, 1])*(-alpha)


def sim_signals(no_sim, len_lc, _x, _y):
    s_ind = np.zeros(no_sim, dtype='int')
    signals = np.zeros((no_sim ,len_lc)) 

    inner = np.arange(_x*_y).reshape(_x, _y)
    inner = vectorize(inner[1:_x-1, 1:_y-1]) # Avoid injecting edge cells

    for i in range(no_sim):
        if i%3 == 0:
            ampl = random.uniform(1, 3) #random.randint(1, 3)
            freq = random.uniform(4, 8)
            sig = ampl*0.005*np.sin(np.arange(len_lc) * (2*pi*freq/len_lc) )
        if i%3 == 1:
            period = random.randint(24*2*4, len_lc//4)
            epoch = random.randint(20, period)
            dur = random.randint(8, 30)
            transit_depth = random.randint(1, 8)*0.002
            sig = soft_transit(len_lc, period, epoch, 40, transit_depth)
        if i%3 == 2:
            sig = vec_shock(len_lc)*0.001
        r = random.choice(inner)
        s_ind[i] = r
        signals[i] = sig
    return s_ind, signals

