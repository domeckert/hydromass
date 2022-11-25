import pymc as pm
import numpy as np

from .deproject import MyDeprojVol
from .functions import rho_nfw_cr


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
    return dsigma


def get_shear(sigma, dsigma, mean_sigm_crit_inv, fl):
    # computes the tangential shear g+ given the mass profile of the cluster (sigma, dsigma) and geometrical
    # situation of background sources(mean_sigm_crit_inv, fl)
    return dsigma * (mean_sigm_crit_inv + fl * sigma * mean_sigm_crit_inv ** 2)


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
    dsigma = dsigma_trap_np(sig, radplus)
    gplus = get_shear(sig, dsigma, WLdata.msigmacrit, WLdata.fl)
    return gplus, rm, ev