import numpy as np
import pickle
import sys
from spatial_detrend.methods.util import *
from spatial_detrend.preproc.kepler_util import module_map
import pkg_resources

quarter = 6

# This file will grid the data into dimensions (cells_per_module * modules_x, cells_per_module * modules_y)
data_dir = pkg_resources.resource_filename('spatial_detrend', 'data/')
position = pickle.load( open(data_dir+"position_%s.p" % (quarter), "rb" ))
sort = pickle.load( open(data_dir+"2sort_%s.p" % (quarter), "rb" ))
position = np.array(position)

#dimensions ~(2200*5) x (2200*5) Each CCD 2200x1024
#position = [row, col, glob_row, glob_col, module, output] 

#=============================================================

# Specific code for the layout/position information
cells_per_module = 10 # discretization of output module into (cells_per_module x cells_per_module)
pixels_per_module = 2200
row_in = pixels_per_module*3
col_in = pixels_per_module*5
_y = 3*cells_per_module #row_out
_x = 5*cells_per_module #col_out
modules_x = 5
modules_y = 3
pixels_per_cell = int(pixels_per_module/cells_per_module) 

# Create cell mapping
map_cell_lc = np.zeros((_y, _x), dtype='int')

sort_pos = np.copy(position[sort,2:4])
loc = np.array([[np.floor_divide(p[0] , pixels_per_cell), np.floor_divide(p[1], pixels_per_cell)] for p in sort_pos]) 

for i in range(_y):
    for j in range(_x):
        co = np.array([i+cells_per_module, j]) # first index is +cells_per_module to account for offset row (_y) (i.e. modules between 2, 3, 4 )
        indices = np.where(np.all(loc == co, axis=1))[0]
        if indices.size != 0: 
            map_cell_lc[i,j] = indices[0] 
        else:
            distance = np.linalg.norm(sort_pos - co*pixels_per_cell, axis=1)
#            distance = np.array([np.sum(np.abs(p-co*pixels_per_cell)) for p in sort_pos]) #older deprecated version
            map_cell_lc[i, j] = np.nanargmin(distance)
        sort_pos[map_cell_lc[i, j], :] = np.array([np.nan, np.nan])
        loc[map_cell_lc[i, j], :] = np.array([np.nan, np.nan])

indices = sort[vectorize(map_cell_lc)]
pickle.dump((indices, _x, _y, modules_x, modules_y), open( data_dir+"grid_indices_%s.p" % (quarter), "wb" ) )

