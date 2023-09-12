import numpy as np
import pickle
from os import listdir
from os.path import isfile, join
from spatial_detrend.methods.util import *
from spatial_detrend.preproc.kepler_util import open_lc_data
import pkg_resources

# USAGE:
# Download lightcurves of interest for a specific quarter from MAST archive and put in folder 'q{quarter}_data'
# Alternatively download through lightkurve/will automate this in the future
# Have included lc_data I made earlier for 3 quarters (6, 10, 14) and mag band (12-13)

quarter = 6

data_dir = pkg_resources.resource_filename('spatial_detrend', 'data/')
#lc_files = [f for f in listdir(data_dir+'q%s_data' % (quarter)) if isfile(join(data_dir+'q%s_data' % (quarter), f))]
#lc_data, quality_flags, pos, k_id = open_lc_data(lc_files, quarter)
#lc_data = np.array([nan_linear_gapfill(median_normal(lc)) for lc in lc_data])

#pickle.dump(lc_data, open(data_dir+'lc_data_filled%s.p' % (quarter), "wb" ) )
#pickle.dump(k_id, open(data_dir+'kid_%s.p' % (quarter), "wb" ) )
#pickle.dump(pos, open(data_dir+'position_%s.p' % (quarter), "wb" ) )

lc_data = pickle.load( open(data_dir+'lc_data_filled%s.p' % (quarter), "rb" ))

#HANDPICK THESE POINTS TO FILTER, these are relevant for Q 10
if quarter == 10:
    lc_data = lc_data[:, 1:]
    outlier_cad = np.array([1127, 1420, 1916, 2929])
    lc_data[:, outlier_cad] = lc_data[:, outlier_cad-1]

numlc = len(lc_data)
len_lc = len(lc_data[0])

#===============
# remove linear trend + normalize

high_f_lc = np.array([mag_normal_med(linear_detrend(lc)) for lc in lc_data])

#=================
# remove outliers by coarse detrending and thresholding

coarse_detrend = PCA_detrend(high_f_lc, 20)

for i in range(numlc):
    high_f_lc[i] = threshold_data(high_f_lc[i], coarse_detrend[i], level=4)

#============================================
# obtains list of highly correlated and variance thresholded lightcurves

sort = filter_lightcurves(high_f_lc)

#=============================================

pickle.dump(sort, open( data_dir + "sort_%s.p" % (quarter), "wb" ) )
pickle.dump(high_f_lc, open( data_dir + "high_f_lc%s.p" % (quarter), "wb" ) )


