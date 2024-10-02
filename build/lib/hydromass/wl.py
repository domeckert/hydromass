import pymc as pm
import numpy as np
import random 
from tqdm import tqdm
from .deproject import MyDeprojVol
from astropy import constants as const
from astropy import units as u

def rho_to_sigma(radii_bins, rho):
    # computes the projected mass density sigma given a density profile rho
    # radii_bins*=1e-6
    deproj = MyDeprojVol(radii_bins[:-1], radii_bins[1:])
    proj_vol = deproj.deproj_vol().T
    area_proj = np.pi * (-(radii_bins[:-1] * 1e6) ** 2 + (radii_bins[1:] * 1e6) ** 2)
    sigma = pm.math.dot(proj_vol, rho) / area_proj
    return sigma * 1e12  # to get result in M_sun * Mpc**-2

def rho_to_sigma_np(radii_bins, rho):
    # computes the projected mass density sigma given a density profile rho
    # radii_bins*=1e-6
    deproj = MyDeprojVol(radii_bins[:-1], radii_bins[1:])
    proj_vol = deproj.deproj_vol().T
    area_proj = np.pi * (-(radii_bins[:-1] * 1e6) ** 2 + (radii_bins[1:] * 1e6) ** 2)
    sigma = np.dot(proj_vol, rho) / area_proj
    return sigma * 1e12  # to get result in M_sun * Mpc**-2


def dsigma_trap(sigma, radii):
    # computes dsigma using numerical trap intergration
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
    # computes dsigma using numerical trap intergration
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
    # computes the tangential shear g+ given the mass profile of the cluster (sigma, dsigma) and geometrical
    # situation of background sources(mean_sigm_crit_inv, fl)

    shear = (dsigma * mean_sigm_crit_inv)/(1 - fl * sigma * mean_sigm_crit_inv)

    return shear



def get_radplus(radii, rmin=1e-3, rmax=1e2, nptplus=19):
    # for the numerical integration to be successful, it is useful to create a set of fictive points at low radii 
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
    radplus, rm, ev = get_radplus(WLdata.radii_wl)
    rho_out = model.rho_pm(radplus, *pmod) * WLdata.rho_crit
    sig = rho_to_sigma(radplus, rho_out)
    dsigma = dsigma_trap(sig, radplus)
    gplus = get_shear(sig, dsigma, WLdata.msigmacrit, WLdata.fl)
    return gplus, rm, ev


def WLmodel_np(WLdata, model, pmod):
    radplus, rm, ev = get_radplus(WLdata.radii_wl)
    rho_out = model.rho_np(radplus, *pmod) * WLdata.rho_crit
    sig = rho_to_sigma_np(radplus, rho_out)
    _, dsigma = dsigma_trap_np(sig, radplus)
    gplus = get_shear(sig, dsigma, WLdata.msigmacrit, WLdata.fl)
    return gplus, rm, ev

def WLmodel_profiles_np(WLdata, model, pmod, rmin, rmax, npt):
    radplus, rm, ev = get_radplus(WLdata.radii_wl, rmin, rmax, npt)
    rho_out = model.rho_np(radplus, *pmod) * WLdata.rho_crit
    sig = rho_to_sigma_np(radplus, rho_out)
    sigbar, _ = dsigma_trap_np(sig, radplus)
    return rho_out, sig, sigbar, rm, ev

def sigbar_envelope_hires(tmhyd, wldata, model, ndraws=500, rmin=1e-3, rmax=1e1, npt=100):
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
    # Compute sigma critical
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