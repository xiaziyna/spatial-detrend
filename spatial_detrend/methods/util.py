import numpy as np
from numpy.random import rand
import more_itertools as mit
import random 
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter

def vectorize(matr):
    """
    Vectorize a 2D matrix in column-major (Fortran-like) order.
    
    Args:
    matr : 2D array Matrix to be vectorized.
        
    Returns:
    1D array Column-major vectorized form of `matr`.
    """
    return np.ravel(matr, order='F')


def PCA_detrend(data, k):
    """
    Perform PCA detrending with rank K on data (a collection of lightcurves)

    Args:
    data: Shape (number of lightcurves, N) where N is the length of a lightcurve.
    k : model rank

    Returns:
    PCA detrended (and mean-subtracted) data.
    """
    pca = PCA(n_components=k)
    pca.fit(data)
    evecs = pca.components_
    mean_subtract_data = data - np.mean(data, axis=0)
    return mean_subtract_data - mean_subtract_data.dot(evecs.T).dot(evecs)

def linear_detrend(lightcurve):
    """
    Linearly detrend an individual lightcurve

    Args:
    lightcurve : Length N time series
    
    Returns:
    lightcurve with linear trend removed
    
    """
    z = np.polyfit(range(len(lightcurve)), lightcurve, 1)
    p = np.poly1d(z)
    return lightcurve - p(range(len(lightcurve)))

def threshold_data(data, base_data = None, level=4):
    """
    Threshold outliers (flux samples) at level*std dev and replace with Gaussian random samples.
    
    The filtering is performed in a two-step procedure by first applying a coarse threshold
    to remove extremal values and using this data to calculate the std dev. 
    
    Args:
    data : 1D array containing the data to be thresholded.
    base_data (optional) : base the calculation of points to be thresholded on base_data if supplied (thresholding applied to data)
    level (optional) : Factor by which the standard deviation is multiplied to set the threshold level. Default is 5.
        
    Returns:
    1D array containing the thresholded data.
        
    """
    if base_data is None: base_data = data
    std_ = np.nanstd(base_data)
    diff = np.diff(base_data, prepend=base_data[0])
    thresh = level*std_
    mask = np.ones(len(base_data), dtype=bool)

    mask[np.abs(base_data) > thresh] = False
    mask[np.abs(diff) > thresh] = False

    std_clean = np.nanstd(base_data[mask])
    thresh = level*std_clean

    mask = np.zeros(len(data), dtype=bool)    
    mask[np.abs(base_data) > thresh] = True
    mask[np.abs(diff) > thresh] = True

    data[mask] = np.random.normal(0, std_clean, size=mask.sum())
    return data

def nan_linear_gapfill(data):
    """
    Fill NaN gaps in data using linear interpolation.
    
    The function identifies groups of consecutive NaNs in the data and fills them using 
    a linear interpolation approach based on the values immediately adjacent to the gaps.
    
    Args:
    data : 1D array containing the data with NaN gaps to be filled.
        
    Returns:
    1D array where NaN gaps have been filled using linear interpolation.
        
    """
    goodind = np.where(~np.isnan(data))
    badind = np.where(np.isnan(data))
    gaps = [list(group) for group in mit.consecutive_groups(badind[0])]
    for g in gaps:
        if len(g) == 1:
            data[g[0]] = data[g[0]-1]
            continue
        else:
            grad = (data[g[len(g)-1]+1]-data[g[0]-1])/(len(g)+2)
            data[g] = (np.arange(len(g))*grad) + data[g[0]-1]
    return data

def filter_lightcurves(data, var_perc = 90, diff_perc = 90, lc_corr_thresh = .6):
    """
    Threshold filter a collection of lightcurves to obtain a nicely behaved set
    # Filter a collection of lightcurves by variances, 1d difference and minimum correlation between lightcurves
    # Must satisfy all of the 3 conditions:
    #1) Obtain the variance of each lightcurve, must be within 90th percentile
    #2) Obtain the mean of the 1d difference of lightcurve mean(|lc[n] - lc[n+1]|), must be within 90th percentile
    #3) Retain lightcurves which have a minimum correlation lc_corr_thresh with at least 10 other lightcurves

    #Returns indices of retained lightcurves 
    """
    var_ = [np.nanvar(lc) for lc in data]
    diff_mean = [np.nanmean(np.abs(np.ediff1d(lc))) for lc in data]
    var_thresh = np.percentile(var_, var_perc)
    diff_mean_thresh = np.percentile(diff_mean, diff_perc)
    numlc = len(data)

    norm_lc = [mag_normal_med(lc) for lc in data]
    corr_sort = np.zeros(numlc)

    for i in range(numlc):
        corr = 0
        for j in range(numlc):
            if j != i:
                if np.inner(norm_lc[i], norm_lc[j]) > lc_corr_thresh: corr+=1
        if corr > 10: corr_sort[i] = 1

    sort = np.logical_and(var_<var_thresh, diff_mean<diff_mean_thresh)
    sort = np.logical_and(sort, corr_sort)
    return np.where(sort)[0]

def corr_comp(data):
    """
    Compute the average pairwise correlation between normalized data vectors.
    
    The function calculates the inner product between each unique pair of 
    elements in the data array and returns the average.
    
    Args:
    data : 2D array Array containing the vectors over which pairwise corr. are calculated.
           Each vector in data must be normalized before calling this function
        
    Returns:
    Average pairwise correlation of data elements.
    """
    n = len(data)
    total_sum = 0
    total_count = 0
    
    for i in range(n):
        for j in range(i+1, n): 
            total_sum += np.inner(data[i], data[j])
            total_count += 1
    
    return total_sum / total_count

def corr_comp_old(data):
    """
    Compute the average pairwise correlation between data elements.
    
    The function calculates the inner product between each unique pair of 
    elements in the data array and returns the average.
    
    Args:
    data : 2D array Array containing the vectors over which pairwise corr. are calculated.
        
    Returns:
    Average pairwise correlation of data elements.
    """
    corr_vals = []
    for m in range(len(data)):
        for l in range(len(data)):
            if m!=l: corr_vals.append(np.inner(data[m], data[l]))
    corr_vals = np.array(corr_vals)
    return np.nanmean(corr_vals)

def simple_corr(x, y):
    """
    Compute the normalized inner product (cosine similarity) between two vectors.
    Slow if computing over many x and y pairs, in this case normalize each x and use inner product.

    Args:
    x, y : Vectors between which the cosine similarity is to be computed.
        
    Returns:
    Cosine similarity between `x` and `y`.
    """
    return np.inner(x,y)/(np.linalg.norm(x) * np.linalg.norm(y))

def median_normal(lightcurve):
    """
    Median normalize lightcurve
    
    Args:
    lightcurve : lightcurve to be normalized.
        
    Returns:
    Median-normalized lightcurve.
    """
    lightcurve -= np.nanmedian(lightcurve)
    return lightcurve / np.nanmedian(np.abs(lightcurve))

def mag_normal_med(lightcurve):
    """
    Normalize lightcurve by magnitude (to be used if calculating pairwise correlation with corr_comp)
    
    Args:
    lightcurve :lightcurve to be normalized.
        
    Returns:
    magnitude-normalized data.
    """
    lightcurve -= np.nanmedian(lightcurve) #Subtract median as slightly more robust to outliers
    return lightcurve / np.linalg.norm(lightcurve)

def mag_normal_mean(lightcurve):
    """
    Normalize lightcurve by magnitude (to be used if calculating pairwise correlation with corr_comp)
    
    Args:
    lightcurve :lightcurve to be normalized.
        
    Returns:
    magnitude-normalized data.
    """
    lightcurve -= np.nanmean(lightcurve)
    return lightcurve / np.linalg.norm(lightcurve)

def calc_CDPP(lightcurve, scale, offset):
    """
    CDPP in PPM
    """
    lightcurve *= scale
    smooth_reg = lightcurve - savgol_filter(lightcurve, 97, 2)
    smooth_reg = threshold_data(smooth_reg)
    mean_bin = np.zeros(len(smooth_reg)-14)
    for j in range(len(mean_bin)):
        mean_bin[j] = np.mean(smooth_reg[j:j+13])
    cdpp_reg = ( np.std(mean_bin)*1.168 / np.median(offset) ) / (1e-6) # CDPP in PPM
    return cdpp_reg
