# spatial-detrend

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Overview

Developed by Jamila Taaki (UIUC).

spatial-detrend is a Python library for detrending collections of lightcurves, using spatial (on the sensor) correlations of systematic/instrument noise. 
The spatial detrending method is described publication 'Robust Detrending of Spatially Correlated Systematics in Kepler Light Curves Using Low-Rank Methods'.
The detrending method is built around a low-rank linear model that's conditioned on a total-variation spatial constraint. 
This constraint fundamentally models spatial systematic structure across the sensor, offering a robust, data-driven solution for systematics mitigation.

This library is currently in an experimental stage and has been tailored for specific use-cases as detailed in our accompanying Astrophysical Journal publication. 
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

If wish to start from scratch, download a collection of lightcurves from a single quarter

download data
preprocess data
generate weight matrices/difference operators
call solver

### Input data

Quarter 6 prepped data included for your convenience. 
For more prepped data see github repo! 

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
    │   ├── cal_flux_10.p
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
If you find this package useful, please cite our Astrophysical Journal paper:

## License

[spatial-detrend] is released under the [GNU General Public License v3.0](LICENSE).

