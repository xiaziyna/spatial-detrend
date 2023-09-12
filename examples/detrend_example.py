import numpy as np
import pickle
from sklearn.decomposition import PCA
from spatial_detrend.methods.solve.solver_weights import gen_W
from spatial_detrend.methods.solve.solver import Solver
from spatial_detrend.methods.util import calc_CDPP
#import matplotlib.pyplot as plt
#from method.simulate.sim_signal import sim_signals
import pkg_resources

# Kepler observational quarter (data for 6, 10, 14 included)
quarter = 6

data_dir = pkg_resources.resource_filename('spatial_detrend', 'data/')
high_f_lc = pickle.load( open(data_dir+"high_f_lc%s.p" % (quarter), "rb" ))
position = pickle.load( open(data_dir+"position_%s.p" % (quarter), "rb" ))
grid_indices, _x, _y, modules_x, modules_y = pickle.load( open(data_dir+"grid_indices_%s.p" % (quarter), "rb" ))
k_id = pickle.load( open(data_dir+"kid_%s.p" % (quarter), "rb" ))

_Y_T = high_f_lc[grid_indices]
_Y_T -= np.mean(_Y_T, axis=0)

# For simulating and injecting astrophysical signals uncomment below:
#s_ind, signals = sim_signals(9, len_lc, _x, _y)
#_Y_T[s_ind] += signals

# Initialize spatial systematics algorithm with PCA coefficients
k = 20
solver_type = 2
pca = PCA(n_components=(k))
pca.fit(_Y_T)
evecs = pca.components_
C_init = evecs[:k].dot(_Y_T.T)

# PCA detrended lightcurves for comparison
detrend_PCA = _Y_T - _Y_T.dot(evecs.T).dot(evecs)

# Generate weight matrices Wx, Wy
corr_xy_x, corr_xy_y, corr_full_x, corr_full_y, corr_modfull_x, corr_modfull_y = gen_W(_Y_T, _x, _y, modules_x, modules_y, solver_type)

# Instantiate Solver class with dimensions
solver1 = Solver(_x, _y, solver_type)

# Generate Difference matrices Dxx, Dyy
solver1.generate_D(Wx = corr_xy_x, Wy = corr_xy_y)
#solver1.generate_D(Wx = corr_full_x, Wy = corr_full_y) 

# Estimate systematic coefficients
C = solver1.solve(k, _Y_T, C_init, p=1.2, alpha=0.2, eta=.1, beta=1e-16, niter=50)

# Estimate systematics
L_T = solver1.est_syst(C, _Y_T)

# Detrend lightcurve
detrend_SP = _Y_T - L_T

# Uncomment to visualize spatial coefficients
#for i in range(k):
#    im = np.reshape(C[i], (_x, _y))
#    plt.figure()
#    plt.imshow(im, cmap = 'PuOr', interpolation='none', vmin=-1, vmax=1)
#    plt.show()

# Uncomment to visualize PCA coefficients
#for i in range(k):
#    im = np.reshape(C_init[i], (_x, _y))
#    plt.figure()
#    plt.imshow(im, cmap = 'PuOr', interpolation='none', vmin=-1, vmax=1)
#    plt.show()

# To rescale lightcurve to original flux/calc CDPP, uncomment lines below
# scale, offset = pickle.load( open( data_dir+"cal_flux_%s.p" % (quarter), "rb" ))
# scale[grid_indices]*detrend_SP + offset[grid_indices]
# print (calc_CDPP(detrend_SP[i], scale[grid_indices][i], offset[grid_indices][i]))
