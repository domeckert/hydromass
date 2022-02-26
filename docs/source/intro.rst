About hydromass
===============

``hydromass`` is a Python package for the analysis of galaxy cluster mass profiles from X-ray and/or Sunyaev-Zel'dovich observations. The code builds upon several decades of development and tens of scientific papers. The code was originally developed as an IDL script by Stefano Ettori (see `Ettori et al. 2010 <https://ui.adsabs.harvard.edu/abs/2010A%26A...524A..68E/abstract>`_ ) and translated into Python by Vittorio Ghirardini during his PhD. The new Python code was completely revised by Dominique Eckert in the framework of the X-COP gravitational field project and turned into a general framework for the reconstruction of galaxy cluster mass and thermodynamic profiles. The code is released together with a series of two papers describing it in detail and applying it to the X-COP galaxy cluster sample.


Motivation
**********

The X-COP collaboration is committed to delivering high-level data products and advanced analysis tools to allow for an easy replication of our results and extension of our work to a wider range of applications than can be pursued only with our limited manpower. The ``hydromass`` package is an important part of this philosophy, as it will allow the user to easily load the public X-COP data products and apply our reconstruction tools directly within a Jupyter notebook. The framework will later be extended to the use of new data such as the CHEX-MATE Heritage program on XMM-Newton and eventually to the data of upcoming missions (e.g. eROSITA, ATHENA).

Features
********

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
