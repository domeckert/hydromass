# hydromass
A Python package for the reconstruction of hydrostatic mass profiles and deprojection of thermodynamic profile from X-ray and/or Sunyaev-Zeldovich data. The package provides a global Bayesian framework for deprojection and mass profile reconstruction, including mass model fitting, forward fitting with parametric and polytropic models, and non-parametric log-normal mixture reconstruction. 

Extensive documentation for the project can be accessed here:

https://hydromass.readthedocs.io/en/latest/index.html

## Features

- Joint modeling of X-ray surface brightness, X-ray spectroscopic temperature, and SZ pressure
- A global framework for mass modeling, deprojection and PSF deconvolution of thermodynamic gas profiles
- Efficient Bayesian optimization based on Hamiltonian Monte Carlo using PyMC3
- Parametric mass model reconstruction including Navarro-Frenk-White, Einasto and several other popular mass models, with automatic or custom priors
- Decomposition of the hydrostatic mass profile into baryonic and dark matter components
- Non-parametric temperature deprojection and hydrostatic mass reconstruction using a log-normal mixture model
- Parametric forward model fitting and effective polytropic reconstruction
- Non-thermal pressure modeling and marginalization
- Diagnostic tools to investigate goodness-of-fit through posterior predictive checks and WAIC
- Easy visualization of the output mass and thermodynamic profiles
- Saving/reloading options

The current implementation has been developed in Python 3 and tested on Python 3.6+ under Linux and Mac OS.

