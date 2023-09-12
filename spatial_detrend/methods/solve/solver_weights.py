import numpy as np
from spatial_detrend.methods.util import vectorize
from spatial_detrend.methods.util import mag_normal_mean

def gen_W(_Y_T, _x, _y, modules_x, modules_y, solver_type = 1):
    """
    Generate various weight matrices to weight total variation prior
    These are passed to the solver class when generating difference matrices Dx and Dy with vp_D 
    TV_W(C) : |W D C|_p

    Args: 
    _Y_T : Lightcurves of shape (X*Y, N)
    _x, _y: Grid dimensions (in terms of cells)
    modules_x, modules_y : Modules along either axis
    solver (1 or 2) : solver 1 (L_2,p) and 2 (L_p) are of different forms, and require different sized weight matrices
    If solver == 1: Wx, Wy ~ (_y, _x)
    If solver == 2: Wx ~ (_y, _x-1), Wy ~ (_y-1, _x)
    """
    cells_per_module = _x // modules_x
    norm_Y_T = np.array([mag_normal_mean(lightcurve) for lightcurve in _Y_T])

    corr_full_x = np.zeros((_y, _x-1))
    for i in range(_y):
        for j in range(_x-1):
            corr_full_x[i, j] = np.inner(norm_Y_T[i+(j*_y)], norm_Y_T[i+((j+1)*_y)])
            if j != 0: corr_full_x[i, j] = (corr_full_x[i, j] + np.inner(norm_Y_T[i+(j*_y)], norm_Y_T[i+((j-1)*_y)]))*.5

    corr_full_y = np.zeros((_x, _y-1))
    for i in range(_x):
        for j in range(_y-1):
            corr_full_y[i, j] = np.inner(norm_Y_T[j+(i*_y)], norm_Y_T[j+1+(i*_y)])
            if j != 0: corr_full_y[i, j] = (corr_full_y[i, j] + np.inner(norm_Y_T[j+(i*_y)], norm_Y_T[j-1+(i*_y)]))*.5

    corr_full_x[corr_full_x < 0] = 0
    corr_full_y[corr_full_y < 0] = 0

    corr_x = np.mean(corr_full_x, axis=1)
    corr_y = np.mean(corr_full_y, axis=1)

    corr_xy = np.zeros((_y, _x)) # average neighbour correlation per cell
    corr_xy[:, :_x-1] += .5 * corr_full_x
    corr_xy[:_y-1, :] += .5 * corr_full_y.T

    #======================================= within module correlation

    loc_map = np.arange(_x*_y)
    loc_map = np.reshape(loc_map, (_x, _y)).T

    mod_corr_y = np.zeros((modules_y, modules_x))
    mod_corr_x = np.zeros((modules_y, modules_x))

    for i in range(modules_y):
        for j in range(modules_x):
            block = vectorize(loc_map[i*cells_per_module:(i+1)*cells_per_module, j*cells_per_module:(j+1)*cells_per_module])
            lc_mod = norm_Y_T[block]
            if i != (modules_x - 1): mod_corr_y[i, j] = np.mean(corr_full_y[j*cells_per_module:(j+1)*cells_per_module, i*cells_per_module:(i+1)*cells_per_module])
            if i == (modules_x - 1): mod_corr_y[i, j] = np.mean(corr_full_y[j*cells_per_module:(j+1)*cells_per_module, i*cells_per_module:])
            if j != (modules_y - 1): mod_corr_x[i, j] = np.mean(corr_full_x[i*cells_per_module:(i+1)*cells_per_module, j*cells_per_module:(j+1)*cells_per_module])
            if j == (modules_y - 1): mod_corr_x[i, j] = np.mean(corr_full_x[i*cells_per_module:(i+1)*cells_per_module, j*cells_per_module:])

    modfull_y = np.zeros((_y-1, _x))
    modfull_x = np.zeros((_y, _x-1))
    for i in range(modules_y):
        for j in range(modules_x):
            modfull_y[i*cells_per_module: (i+1)*cells_per_module - 1, j*cells_per_module: (j+1)*cells_per_module] = np.ones((cells_per_module-1, cells_per_module))*mod_corr_y[i, j]
            modfull_x[i*cells_per_module: (i+1)*cells_per_module, j*cells_per_module: (j+1)*cells_per_module - 1] = np.ones((cells_per_module, cells_per_module-1))*mod_corr_x[i, j]

    if solver_type == 1:
        corr_xy_x, corr_xy_y = corr_xy, corr_xy
        modfull_x = np.append(modfull_x, np.zeros((_y, 1)), axis=1)
        modfull_y = np.append(modfull_y, np.zeros((1, _x)), axis=0)
        corr_full_x = np.append(corr_full_x, np.zeros((_y, 1)), axis=1)       
        corr_full_y = np.append(corr_full_y.T, np.zeros((1, _x)), axis=0)

    if solver_type == 2:
        corr_xy_x = corr_xy[:, :_x-1]
        corr_xy_y = corr_xy[:_y-1, :]

    # Naming convention *x, *y denotes whether Wx or Wy
    # corr_modfull is a weighting by the average within module correlation (directionally in x or y)
    # corr_full is the directional correlation per cell (i.e. neighbour correlation along x or y)
    # Average correlation per module
    corr_modfull_x = vectorize(modfull_x)
    corr_modfull_y = vectorize(modfull_y) #type 2

    # Correlation per cell, per axis
    corr_full_x = vectorize(corr_full_x)
    corr_full_y = vectorize(corr_full_y.T) #type 2
    corr_xy_x = vectorize(corr_xy_x) 
    corr_xy_y = vectorize(corr_xy_y)
    return corr_xy_x, corr_xy_y, corr_full_x, corr_full_y, corr_modfull_x, corr_modfull_y
