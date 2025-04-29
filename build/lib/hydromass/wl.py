import pymc as pm
import numpy as np
from astropy import constants as const
from astropy import units as u
import random 
from tqdm import tqdm
from .deproject import *


def rho_to_sigma(radii_bins, rho):
    """
    Theano function. Computes the projected mass density sigma given a density profile rho.
    :param radii_bins: array-like, radii bins
    :param rho: array-like, density profile
    return: sigma, array-like, surface mass density [M_sun * Mpc**-2]
    """

    deproj = MyDeprojVol(radii_bins[:-1], radii_bins[1:])
    proj_vol = deproj.deproj_vol().T
    area_proj = np.pi * (-(radii_bins[:-1] * 1e6) ** 2 + (radii_bins[1:] * 1e6) ** 2)
    sigma = pm.math.dot(proj_vol, rho) / area_proj
    return sigma * 1e12  # to get result in M_sun * Mpc**-2

def rho_to_sigma_np(radii_bins, rho):
    """
    Numpy function. Computes the projected mass density sigma given a density profile rho.
    :param radii_bins: array-like, radii bins
    :param rho: array-like, density profile
    return: sigma, array-like, surface mass density [M_sun * Mpc**-2]
    """

    deproj = MyDeprojVol(radii_bins[:-1], radii_bins[1:])
    proj_vol = deproj.deproj_vol().T
    area_proj = np.pi * (-(radii_bins[:-1] * 1e6) ** 2 + (radii_bins[1:] * 1e6) ** 2)
    sigma = np.dot(proj_vol, rho) / area_proj
    return sigma * 1e12  # to get result in M_sun * Mpc**-2


def dsigma_trap(sigma, radii):
    """
    Theano function. Computes dsigma using numerical trap intergration 
    :param sigma: array-like, surface mass density
    :param radii: array-like, radii bins
    return: dsigma, array-like, excess surface mass density [M_sun * Mpc**-2]
    """
    rmean = (radii[1:] + radii[:-1]) / 2
    rmean2 = (rmean[1:] + rmean[:-1]) / 2
    m = np.tril(np.ones((len(rmean2) + 1, len(rmean2) + 1)))
    dr = rmean[1:] - rmean[:-1]

    ndr = len(dr)

    arg0 = sigma[0] * (rmean2[0] ** 2)/2
    arg1 = dr * (sigma[1:] * rmean[1:] + sigma[:-1] * rmean[:-1]) / 2

    list_stack = [arg0]

    for i in range(ndr):
        list_stack.append(arg1[i])

    arg = pm.math.stack(list_stack)
    a = pm.math.dot(m, arg)
    sigmabar = (2 / (rmean ** 2)) * a
    dsigma = sigmabar - sigma
    return dsigma

def dsigma_trap_np(sigma, radii):
    """
    Numpy function. Computes excess surface mass density dsigma using numerical trap intergration
    :param sigma: array-like, surface mass density
    :param radii: array-like, radii bins
    return: sigmabar, array-like, excess surface mass density [M_sun * Mpc**-2]
            dsigma, array-like, average surface mass density [M_sun * Mpc**-2]
    """
    rmean = (radii[1:] + radii[:-1]) / 2
    rmean2 = (rmean[1:] + rmean[:-1]) / 2
    m = np.tril(np.ones((len(rmean2) + 1, len(rmean2) + 1)))
    dr = rmean[1:] - rmean[:-1]

    ndr = len(dr)

    arg0 = sigma[0] * (rmean2[0] ** 2)/2
    arg1 = dr * (sigma[1:] * rmean[1:] + sigma[:-1] * rmean[:-1]) / 2

    arg = np.append(arg0, arg1)

    a = np.dot(m, arg)
    sigmabar = (2 / (rmean ** 2)) * a
    dsigma = sigmabar - sigma
    return sigmabar, dsigma


def get_shear(sigma, dsigma, mean_sigm_crit_inv, fl):
    """
    Computes the tangential shear g+ given the mass profile of the cluster (sigma, dsigma) and geometrical situation of background sources(mean_sigm_crit_inv, fl)
    Using the approximation of Seitz and Schneider 1997 (or Umetsu 2020 eq.93) formula.
    :param sigma: array-like, surface mass density
    :param dsigma: array-like, excess surface mass density 
    :param mean_sigm_crit_inv: float, value of the inverse mean critical density <sigcrit**-1> in Mpc**2.Msun**-1
    :param fl: float, value of <sigcrit**-2> / (<sigcrit**-1>**2), useful for 2nd order approximation of the shear.
    return: shear, array-like, tangential shear
    """
    shear = (dsigma * mean_sigm_crit_inv)/(1 - fl * sigma * mean_sigm_crit_inv)

    return shear



def get_radplus(radii, rmin=1e-3, rmax=1e2, nptplus=19):
    """
    Creates a set of radii for the numerical integration of the mass profile by interpolating
    and extrapolating both at low and high radii.
    :param radii: array-like, radii bins
    :param rmin: float, minimum radius
    :param rmax: float, maximum radius
    :param nptplus: int, number for low radii extrapolation and interpolation (high radii extrapolation fixed to 20)
    returns:
        radplus: array-like, radii for the numerical integration
        rmeanplus: array-like, mean radii for the numerical integration
        evalrad: array-like, indices for the evaluation of the mass profile
    """
    if nptplus % 2 == 0:
        nptplus = nptplus + 1
    rmean = (radii[1:] + radii[:-1]) / 2.
    radplus = np.logspace(np.log10(rmin), np.log10(radii[0]), nptplus)
    for i in range(len(radii) - 1):
        vplus = np.linspace(radii[i], radii[i + 1], nptplus + 1)
        radplus = np.append(radplus, vplus[1:])
    radplus = np.append(radplus, np.logspace(np.log10(radplus[-1]), np.log10(rmax), 20)[1:])
    rmeanplus = (radplus[1:] + radplus[:-1]) / 2.
    nsym = int(np.floor(nptplus / 2))
    evalrad = (np.arange(nptplus + nsym - 1, nptplus + nsym + len(rmean) * nptplus, nptplus))[:len(rmean)]
    return radplus, rmeanplus, evalrad


def WLmodel(WLdata, model, pmod):
    """
    Theano function. Computes the tangential shear g+ for a given model and set of parameters.
    :param WLdata: object, weak lensing data
    :param model: object, mass model (e.g. NFW, Einasto)
    :param pmod: array-like, model parameters (e.g. [c200, r200] for NFW)
    return: gplus, array-like, tangential shear
            rm, array-like, mean radii for the numerical integration
            ev, array-like, indices for the evaluation of the mass profile"""
    radplus, rm, ev = get_radplus(WLdata.radii_wl)
    rho_out = model.rho_pm(radplus, *pmod) * WLdata.rho_crit
    sig = rho_to_sigma(radplus, rho_out)
    dsigma = dsigma_trap(sig, radplus)
    gplus = get_shear(sig, dsigma, WLdata.msigmacrit, WLdata.fl)
    return gplus, rm, ev

# def WLmodel_elong(WLdata, model, pmod, elong):
#     """
#     Theano function. Computes the tangential shear g+ for a given model and set of parameters.
#     :param WLdata: object, weak lensing data
#     :param model: object, mass model (e.g. NFW, Einasto)
#     :param pmod: array-like, model parameters (e.g. [c200, r200] for NFW)
#     return: gplus, array-like, tangential shear
#             rm, array-like, mean radii for the numerical integration
#             ev, array-like, indices for the evaluation of the mass profile"""
#     radplus, rm, ev = get_radplus(WLdata.radii_wl)
#     rho_out = model.rho_pm(radplus, *pmod) * WLdata.rho_crit
#     sig = rho_to_sigma(radplus, rho_out)
#     sig_elong = elongation_correction(sig, rm, np.arange((len(rm)))[1:-1], elong)
#     dsigma = dsigma_trap(sig_elong, radplus[1:-1])
#     gplus = get_shear(sig_elong, dsigma, WLdata.msigmacrit, WLdata.fl)
#     return gplus, rm, ev


# def WLmodel_np(WLdata, model, pmod, elong, n_draw=None, random_state=None):
#     """
#     Computes the tangential shear g+ for one or multiple sets of model parameters.
#     Optionally, a random subset of samples can be drawn.

#     :param WLdata: object, weak lensing data
#     :param model: object, mass model (e.g. NFW, Einasto)
#     :param pmod: array-like, either a 1D array of one parameter set (e.g., [c200, r200])
#                  or a 2D array with multiple sets of parameters.
#     :param n_draw: int, optional, number of random samples to draw from pmod.
#                    If None, all samples are used.
#     :param random_state: int or np.random.Generator, optional, seed for reproducibility.
#     :return:
#         gplus_all: 2D array of tangential shear, shape (M, n_draw), where M = len(rm), n_draw = selected parameter sets
#         rm: array-like, mean radii for the numerical integration
#         ev: array-like, indices for the evaluation of the mass profile
#     """
#     radplus, rm, ev = get_radplus(WLdata.radii_wl)
    
#     # Ensure pmod is 2D for consistency
#     pmod = np.atleast_2d(pmod)  # Converts 1D array to 2D if needed
#     n_samples = pmod.shape[0]
    
#     # Select samples if n_draw is specified
#     if n_draw is not None and n_draw < n_samples:
#         rng = np.random.default_rng(random_state)
#         indices = rng.choice(n_samples, n_draw, replace=False)
#         pmod = pmod[indices]
#         n_samples = pmod.shape[0]
    
#     # Initialize the result array
#     gplus_all = np.zeros((len(rm), n_samples))
    
#     # Loop over all parameter sets
#     for i in tqdm(range(n_samples)):
#         rho_out = model.rho_np(radplus, *pmod[i]) * WLdata.rho_crit
#         sig = rho_to_sigma_np(radplus, rho_out)
#         _, dsigma = dsigma_trap_np(sig, radplus)
#         gplus = get_shear(sig, dsigma, WLdata.msigmacrit, WLdata.fl)
        
#         # Extract values at the evaluation radii
#         gplus_all[:, i] = gplus
    
#     return gplus_all, rm, ev


def WLmodel_profiles_np(WLdata, model, pmod, rmin, rmax, npt):
    """
    Numpy function. Computes the mass profile, surface mass density, excess surface mass density, 
    mean radii and indices for the evaluation of the mass profile. 
    Option to directly tune the radial range and number of points for higher radial resolution.
    :param WLdata: object, weak lensing data
    :param model: object, mass model (e.g. NFW, Einasto)
    :param pmod: array-like, model parameters (e.g. [c200, r200] for NFW)
    :param rmin: float, minimum radius
    :param rmax: float, maximum radius
    :param npt: int, number of points for the numerical integration
    return: rho_out, array-like, mass profile
            sig, array-like, surface mass density
            sigbar, array-like, mean surface mass density
            rm, array-like, mean radii for the numerical integration
            ev, array-like, indices for the evaluation of the mass profile"""
    radplus, rm, ev = get_radplus(WLdata.radii_wl, rmin, rmax, npt)
    rho_out = model.rho_np(radplus, *pmod) * WLdata.rho_crit
    sig = rho_to_sigma_np(radplus, rho_out)
    sigbar, _ = dsigma_trap_np(sig, radplus)
    return rho_out, sig, sigbar, rm, ev

def sigbar_envelope_hires(tmhyd, wldata, model, ndraws=500, rmin=1e-3, rmax=1e1, npt=100):
    """
    Computes the mean surface mass density profiles sigbar for all the posterior chain of a previous analysis.
    :param tmhyd: object, hydrostatic mass profile
    :param wldata: object, weak lensing data
    :param model: object, mass model (e.g. NFW, Einasto)
    :param ndraws: int, number of profiles to sample
    :param rmin: float, minimum radius
    :param rmax: float, maximum radius
    :param npt: int, number of points for the numerical integration
    return: sigbar_all, array-like, mean surface mass density profiles
            rm, array-like, mean radii for the numerical integration"""
    
    num_pairs = len(tmhyd.samppar)
    sigbar_arr = []

    # Determine the number of elements to sample
    num_samples = min(ndraws, num_pairs)
    
    # Randomly select num_samples indices from the range of available indices
    selected_indices = random.sample(range(num_pairs), num_samples)

    for i in tqdm(selected_indices, desc=f"Calculating gplus for {tmhyd}", unit="pair"):
        par_values = tmhyd.samppar[i]
        _, _, sigbar, rm, _ = WLmodel_profiles_np(wldata, model, par_values, rmin=rmin, rmax=rmax, npt=npt)

        # Check if tmhyd_xwl_e.elong is a 1D array with the same length as tmhyd.samppar
        if hasattr(tmhyd, 'elong') and isinstance(tmhyd.elong, np.ndarray) and len(tmhyd.elong) == num_pairs:
            sigbar *= tmhyd.elong[i]  

        sigbar_arr.append(sigbar)

    sigbar_all = np.array(sigbar_arr)
    return sigbar_all, rm


def get_einstein_r(tmhyd, wldata, model, z_cl, zs, rmin=1e-3, rmax=1e2, npt=100, ndraws=500):
    """
    Computes the Einstein radius for a given cluster and source redshift by equalizing the average surface mass density
    to the critical surface mass density. Einstein radius having values of the order of the arscecond, it is necessary to
    extrapolate at lower radii and with more points than usual with the get_radplus function.
    :param tmhyd: object, hydrostatic mass profile
    :param wldata: object, weak lensing data
    :param model: object, mass model (e.g. NFW, Einasto)
    :param z_cl: float, cluster redshift
    :param zs: float, source redshift
    :param rmin: float, minimum radius
    :param rmax: float, maximum radius
    :param npt: int, number of points for the numerical integration
    :param ndraws: int, number of profiles to sample
    return: rm, array-like, mean radii for the numerical integration
            sigbar_arr, array-like, mean surface mass density profiles
            einstein_r_median, float, median Einstein radius
            einstein_r_16th, float, 16th percentile Einstein radius
            einstein_r_84th, float, 84th percentile Einstein radius
            sigcrit, float, critical surface mass density [M_sun * Mpc**-2]"""
    # Compute sigma crit
    cosmo = tmhyd.cosmo
    c_mpc = const.c.to(u.Mpc / u.s)
    g_mpc = const.G.to(u.Mpc**3 / (u.kg * u.s**2))
    prefactor_mpc = c_mpc**2 / (4 * np.pi * g_mpc)
    dl = cosmo.angular_diameter_distance(z_cl)
    ds = cosmo.angular_diameter_distance(zs)
    dls = cosmo.angular_diameter_distance_z1z2(z_cl, zs)
    sigcrit = ((prefactor_mpc * ds / (dl * dls)).to(u.M_sun / u.Mpc**2)).value
    
    # Get sigbar profiles and radial distances (rm) from sigbar_envelope_hires
    sigbar_arr, rm = sigbar_envelope_hires(tmhyd, wldata, model, ndraws, rmin, rmax, npt)
    
    # Initialize an array to store Einstein radii for each profile
    einstein_r_arr = []
    
    # Loop over each sigbar profile and find the Einstein radius
    for sigbar in sigbar_arr: 
        # Find the index where |sigbar - sigcrit| is minimized for the current profile
        idx = np.argmin(np.abs(sigbar - sigcrit))
        
        # Extract the corresponding rm value (Einstein radius)
        einstein_r = rm[idx]
        einstein_r_arr.append(einstein_r)
    
    # Convert the list of Einstein radii to a numpy array
    einstein_r_arr = np.array(einstein_r_arr)
    
    # Compute the median and 16th/84th percentiles
    einstein_r_median = np.median(einstein_r_arr)
    einstein_r_16th = np.percentile(einstein_r_arr, 16)
    einstein_r_84th = np.percentile(einstein_r_arr, 84)
    
    return rm, sigbar_arr, einstein_r_median, einstein_r_16th, einstein_r_84th, sigcrit