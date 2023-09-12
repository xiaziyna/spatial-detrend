import numpy as np
from astropy.io import fits
import pkg_resources

module_map = np.array([[0, 2, 3, 4, 0],[6, 7, 8, 9, 10],[11, 12, 13, 14, 15],[16, 17, 18, 19, 20],[0, 22, 23, 24, 0]])
sc = (2200/2048)

def kep_position(row, col, module, output):
    col_map= 0
    row_map= 0
    if module in [2, 3, 4, 7, 8]:
        if output == 1:
            col_map= col
            row_map= (1024 + (1024 - row))*sc
        elif output == 2:
            col_map= 2200 - col
            row_map= (1024 + (1024 - row))*sc
        elif output == 3:
            col_map= 2200 - col
            row_map= row*sc
        elif output == 4:
            col_map= col
            row_map= row*sc
    elif module in [18, 19, 22, 23, 24]:
        if output == 1:
            col_map= 2200 - col
            row_map= row*sc
        elif output == 2:
            col_map= col
            row_map= row*sc
        elif output == 3:
            col_map= col
            row_map= (1024 + (1024 - row))*sc
        elif output == 4:
            col_map= 2200 - col
            row_map= (1024 + (1024 - row))*sc
    elif module in [6, 11, 12, 13, 16, 17]:
        if output == 1:
            col_map= (1024 + (1024-row))*sc
            row_map= 2200 - col
        elif output == 2:
            col_map= (1024 + (1024-row))*sc
            row_map= col
        elif output == 3:
            col_map= row*sc
            row_map= col
        elif output == 4:
            col_map= row*sc
            row_map= 2200 - col
    elif module in [9, 10, 14, 15, 20]:
        if output == 1:
            col_map= row*sc
            row_map= col
        elif output == 2:
            col_map= row*sc
            row_map= 2200 - col
        elif output == 3:
            col_map= (1024 + (1024-row))*sc
            row_map= 2200 - col        
        elif output == 4:
            col_map= (1024 + (1024-row))*sc
            row_map= col
    col_map= max(0, min(col_map, 2200))
    row_map= max(0, min(row_map, 2200))
    row_, col_ = np.where(module_map == int(module))
    row_global = (row_[0]*2200) + row_map
    col_global = (col_[0]*2200) + col_map
    return row_map, col_map, row_global, col_global

def open_lc_data(filenames, quarter):
    numlc = len(onlyfiles)
    lc_data = []
    quality_flags = []
    pos = []
    k_id = []
    
    for i in range(numlc):
        try:
            with fits.open(pkg_resources.resource_filename('spatial_detrend', 'data/q'+str(quarter)+'_data/')+onlyfiles[i], memmap=False, ignore_missing_end=True) as hdulist:
                sapfluxes = hdulist[1].data['SAP_FLUX']
                k_id.append(hdulist[0].header['KEPLERID'])
                lc_data.append(sapfluxes)
                module = hdulist[0].header['MODULE']
                output = hdulist[0].header['OUTPUT']
                quality = hdulist[1].data['SAP_QUALITY']
                quality_flags.append(quality)
                col_vals = hdulist[1].data['MOM_CENTR1']
                row_vals = hdulist[1].data['MOM_CENTR2']
                row, col, row_global, col_global = kep_position(np.nanmean(row_vals)-1, np.nanmean(col_vals)-1, module, output)
                pos.append([row, col, row_global, col_global, module, output])
        except: continue

    lc_data = np.array(lc_data)
    return lc_data, quality_flags, pos, k_id

