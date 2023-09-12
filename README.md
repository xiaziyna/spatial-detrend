# spatial-detrend

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Overview

Developed by Jamila Taaki (UIUC).

spatial-detrend is a Python library for detrending collections of lightcurves, using spatial (on the sensor) correlations of systematic/instrument noise. 
The spatial detrending method is described publication 'Robust Detrending of Spatially Correlated Systematics in Kepler Light Curves Using Low-Rank Methods'.
The detrending method is built around a low-rank linear model that's conditioned on a total-variation spatial constraint. 
This constraint fundamentally models spatial systematic structure across the sensor, offering a robust, data-driven solution for systematics mitigation.

This library is currently in an experimental stage and has been tailored for specific use-cases as detailed in our accompanying Astronomical Journal publication. 
It may not be highly generalizable across all kinds of datasets or astrophysical applications. 

This library is compatible with Python 3.6 and later versions. 

## Example

## Installation

You can install spatial-detrend using pip:

```bash
pip3 install spatial-detrend
```

## Dependencies

Scipy, Numpy, Sklearn, Astropy (if using external data)

### Use

1) Download a collection of Kepler SAP lightcurves from a single quarter ([MAST archive](https://archive.stsci.edu/kepler/data_search/search.php) is one way), put these in a folder i.e. 'q2_data'
   for quarter 2 lightcurves.
   -- If wish to skip this step, prepped data available for quarters (6, 10, 14) for Kepler magnitude (12-13) stars, see note under Input data. 
3) Use `spatial_detrend.preproc.kepler_util.open_lc_data` to extract data. Modify and use the script `preproc/preprocess_data.py`
   to call `open_lc_data` and perform filtering of the data. 
4) Run `preproc/grid_data.py` to obtain a discretized sensor and gridded lightcurves (modify relevant parameters).
5) See the example provided for how to call the spatial detrending method with the `methods.solve.solver` class and choose input parameters.

### Input data

To use prepped data, use [git lfs](https://git-lfs.com): 
```bash
$ git clone https://github.com/xiaziyna/spatial-detrend.git spatial-detrend
$ cd spatial-detrend
$ git lfs pull
```

### Parameters

### Worked examples

See examples folder for a demo. 

## Organization

<pre>
spatial-detrend/
├── examples/
│   └── detrend_example.py
├── README.md
├── setup.py
└── spatial_detrend/
    ├── data/
    │   ├── cal_flux_6.p
    │   ......
    │   └── sort_6.p
    ├── methods/
    │   ├── simulate/
    │   │   └── sim_signal.py
    │   ├── solve/
    │   │   ├── solver.py
    │   │   └── solver_weights.py
    │   └── util.py
    └── preproc/
        ├── grid_data.py
        ├── kepler_util.py
        └── preprocess_data.py

</pre>

## Citation
If you find this package useful, please cite our Astronomical Journal paper:

## License

[spatial-detrend] is released under the [GNU General Public License v3.0](LICENSE).

