from .constants import *
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from .deproject import calc_density_operator, calc_grad_operator, calc_sb_operator, MyDeprojVol
from .plots import rads_more
from astropy.io import fits

def delta_func(r, Mhyd, model, pars):
    """
    Return profile of overdensity Delta with respect to critical density for a given input mass model

    :param r: Radii in kpc
    :type r: numpy.ndarray or float
    :param Mhyd: :class:`hydromass.mhyd.Mhyd` object containing mass reconstruction
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param model: :class:`hydromass.functions.Model` object containing the definition of the mass model
    :type model: class:`hydromass.function.Model`
    :param pars: Parameter vector to be passed to the mass model
    :type pars: numpy.ndarray
    :return: Overdensity as a function of radius
    :rtype: numpy.ndarray
    """

    mass = Mhyd.mfact * model.func_np(r, pars, delta=model.delta) * 1e13 * Msun

    vol = 4. / 3. * np.pi * r ** 3 * cgskpc ** 3

    rhoc = Mhyd.cosmo.critical_density(Mhyd.redshift).value

    return mass / vol / rhoc

def mgas_delta(rdelta, coefs, Mhyd, fit_bkg=False, rout_m=None):
    """
    Compute Mgas at an input radius R_delta

    :param rdelta: R_delta in kpc
    :type rdelta: float
    :param coefs: Coefficients describing the density profile
    :type coefs: numpy.ndarray
    :param Mhyd: :class:`hydromass.mhyd.Mhyd` object containing reconstruction
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param fit_bkg: Set whether the background was jointly fitted (True) or subtracted (False). Defaults to False.
    :type fit_bkg: bool
    :param rout_m: If a radially dependent conversion factor is used, radius grid on which the conversion factors were computed
    :type rout_m: numpy.ndarray
    :return: Gas mass evaluated inside rdelta
    :rtype: numpy.ndarray
    """

    rout = np.logspace(np.log10(Mhyd.sbprof.bins[0] * Mhyd.amin2kpc), np.log10(rdelta), 100)

    rin = np.roll(rout, 1)

    rin[0] = 0.

    Kdens = calc_density_operator(rout / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc, withbkg=fit_bkg)

    if Mhyd.cf_prof is not None and rout_m is not None:

        cfp = np.interp(rout, rout_m, Mhyd.cf_prof)

    else:

        cfp = Mhyd.ccf

    dens = np.sqrt(np.dot(Kdens, np.exp(coefs)) / cfp * Mhyd.transf)

    # Matrix containing integration volumes
    volmat = 4. / 3. * np.pi * (rout ** 3 - rin ** 3)

    # Compute Mgas profile as cumulative sum over the volume

    nhconv = cgsamu * Mhyd.mu_e * cgskpc ** 3 / Msun  # Msun/kpc^3

    mgas_d = np.sum(dens * nhconv * volmat)

    return mgas_d


def mbar_overdens(rmax, coefs, Mhyd, fit_bkg=False, rout_m=None):
    """
    Compute overdensity of baryonic mass with respect to critical

    :param rmax: Maximum radius of Mgas calculation
    :type rmax: float
    :param coefs: Coefficients describing the density profile
    :type coefs: numpy.ndarray
    :param Mhyd: :class:`hydromass.mhyd.Mhyd` object containing reconstruction
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param fit_bkg: Set whether the background was jointly fitted (True) or subtracted (False). Defaults to False.
    :type fit_bkg: bool
    :param rout_m: If a radially dependent conversion factor is used, radius grid on which the conversion factors were computed
    :type rout_m: numpy.ndarray
    :return: Radius, Overdensity of Mgas
    :rtype: numpy.ndarray, numpy.ndarray
    """

    nvalm = 100

    rout = np.logspace(np.log10(Mhyd.sbprof.bins[0] * Mhyd.amin2kpc), np.log10(rmax), nvalm)

    rin = np.roll(rout, 1)

    rin[0] = 0.

    Kdens = calc_density_operator(rout / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc, withbkg=fit_bkg)

    if Mhyd.cf_prof is not None and rout_m is not None:

        cfp = np.interp(rout, rout_m, Mhyd.cf_prof)

    else:

        cfp = Mhyd.ccf

    dens = np.sqrt(np.dot(Kdens, np.exp(coefs)) / cfp * Mhyd.transf)

    # Matrix containing integration volumes
    volmat = 4. / 3. * np.pi * (rout ** 3 - rin ** 3)

    # Compute Mgas profile as cumulative sum over the volume

    nhconv = cgsamu * Mhyd.mu_e * cgskpc ** 3  # g/kpc^3

    ones_mat = np.ones((nvalm, nvalm))

    cs_mat = np.tril(ones_mat)

    mgas = np.dot(cs_mat, dens * nhconv * volmat)

    vol = 4. / 3. * np.pi * rout ** 3 * cgskpc ** 3

    rhoc = Mhyd.cosmo.critical_density(Mhyd.redshift).value

    mgas_ov = mgas / vol / rhoc

    if Mhyd.mstar is not None:

        r_mstar = Mhyd.mstar[:, 0]

        cum_mstar = Mhyd.mstar[:, 1] * Msun

        mstar_m = np.interp(rout, r_mstar, cum_mstar)

        mbar_ov = mgas_ov + mstar_m / vol /rhoc

    else:

        mbar_ov = mgas_ov

    return rout, mbar_ov


def calc_rdelta_mdelta(delta, Mhyd, model, plot=False, r0=500., rmax=4000., thin=10):
    '''
    For a given input overdensity Delta, compute R_delta, M_delta, Mgas_delta, fgas_delta and their uncertainties from a loaded mass model reconstruction

    :param delta: Overdensity with respect to critical
    :type delta: float
    :param Mhyd: :class:`hydromass.mhyd.Mhyd` object containing the results of mass reconstruction run
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param model: :class:`hydromass.functions.Model` object defining the mass model
    :type model: class:`hydromass.functions.Model`
    :param plot: If plot=True, returns a matplotlib.pyplot.figure drawing the mass distribution of the chains at R_delta. In case plot=False the function returns an empty figure.
    :type plot: bool
    :param r0: Initial guess for the overdensity radius (in kpc). Defaults to 500
    :type r0: float
    :param rmax: Maximum radius where to search for the overdensity radius (in kpc). Defaults to 4000
    :type rmax: float
    :param thin: Thinning factor for the calculation. If the number of samples is large the calculation can be long. If thin is greater than 1 then the calculation while be done for fewer samples every "thin" value. Defaults to 10.
    :type thin: int
    :return:  Dictionary containing values of R_delta, M_delta, Mgas_delta, Fgas_delta and their 1-sigma percentiles, and figure if plot=True
    :rtype:
        - dict{12xfloat}
        - matplotlib.pyplot.figure
    '''

    nsamp = int(len(Mhyd.samppar) / thin)


    mdelta, rdelta, mgdelta, fgdelta = np.empty(nsamp), np.empty(nsamp), np.empty(nsamp), np.empty(nsamp)

    rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore=Mhyd.nmore)

    for i in range(nsamp):

        ti = i * thin

        if Mhyd.dmonly:

            r_mbar, mbar_ov = mbar_overdens(rmax, Mhyd.samples[i], Mhyd, fit_bkg = Mhyd.fit_bkg, rout_m=rout_m)

            temp_func = lambda x: (delta_func(np.array([x]), Mhyd, model, np.array([Mhyd.samppar[ti]])) + np.interp(x, r_mbar, mbar_ov) - delta) ** 2

        else:

            temp_func = lambda x: (delta_func(np.array([x]), Mhyd, model, np.array([Mhyd.samppar[ti]])) - delta) ** 2

        res = minimize(temp_func, r0, method='Nelder-Mead')

        rdelta[i] = res['x'][0]

        mdelta[i] = 4. / 3. * np.pi * rdelta[i] ** 3 * cgskpc ** 3 * delta * Mhyd.cosmo.critical_density(Mhyd.redshift).value / Msun

        if not Mhyd.wlonly:

            mgdelta[i] = mgas_delta(rdelta[i], Mhyd.samples[ti], Mhyd, fit_bkg = Mhyd.fit_bkg, rout_m=rout_m)

            fgdelta[i] = mgdelta[i] / mdelta[i]

        else:

            mgdelta[i] = 0.

            fgdelta[i] = 0.

    rd, rdlo, rdhi = np.percentile(rdelta, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.])

    md, mdlo, mdhi = np.percentile(mdelta, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.])

    mgd, mgdlo, mgdhi = np.percentile(mgdelta, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.])

    fgd, fgdlo, fgdhi = np.percentile(fgdelta, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.])

    dict = {
        "R_DELTA": rd,
        "R_DELTA_LO": rdlo,
        "R_DELTA_HI": rdhi,
        "M_DELTA": md,
        "M_DELTA_LO": mdlo,
        "M_DELTA_HI": mdhi,
        "MGAS_DELTA": mgd,
        "MGAS_DELTA_LO": mgdlo,
        "MGAS_DELTA_HI": mgdhi,
        "FGAS_DELTA": fgd,
        "FGAS_DELTA_LO": fgdlo,
        "FGAS_DELTA_HI": fgdhi
    }

    if plot:

        plt.clf()

        fig = plt.figure(figsize=(13, 10))

        ax_size = [0.14, 0.12,
                   0.85, 0.85]

        ax = fig.add_axes(ax_size)

        ax.minorticks_on()

        ax.tick_params(length=20, width=1, which='major', direction='in', right=True, top=True)

        ax.tick_params(length=10, width=1, which='minor', direction='in', right=True, top=True)

        for item in (ax.get_xticklabels() + ax.get_yticklabels()):

            item.set_fontsize(22)

        plt.hist(mdelta, bins=30, density=True)

        plt.xlabel('$M_{\Delta} [M_\odot]$', fontsize=40)

        plt.ylabel('Frequency', fontsize=40)

        return  dict, fig

    else:

        return dict


from scipy.optimize import brentq


def calc_rdelta_mdelta_GP(delta, Mhyd, plot=False, r0=500., thin=10):
    '''
    For a given input overdensity Delta, compute R_delta, M_delta, Mgas_delta, fgas_delta and their uncertainties from a loaded non-parametric GP reconstruction

    :param delta: Overdensity with respect to critical
    :type delta: float
    :param Mhyd: :class:`hydromass.mhyd.Mhyd` object containing the results of mass reconstruction run
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param plot: If plot=True, returns a matplotlib.pyplot.figure drawing the mass distribution of the chains at R_delta. In case plot=False the function returns an empty figure.
    :type plot: bool
    :param r0: Initial guess for the overdensity radius (in kpc). Defaults to 500
    :type r0: float
    :return:  Dictionary containing values R_delta, M_delta, Mgas_delta, Fgas_delta and their 1-sigma percentiles, figure if plot=True
    :rtype:
        - dict{12xfloat}
        - matplotlib.pyplot.figure
    '''

    nsamp = int(len(Mhyd.samppar) / thin)

    mdelta, rdelta, mgdelta, fgdelta = np.empty(nsamp), np.empty(nsamp), np.empty(nsamp), np.empty(nsamp)

    rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore=Mhyd.nmore)

    nvalm = len(rin_m)

    nori = len(Mhyd.samppar)

    if Mhyd.cf_prof is not None:

        cf_prof = np.repeat(Mhyd.cf_prof, nsamp).reshape(nvalm, nori)

    else:

        cf_prof = Mhyd.ccf

    dens_m = np.sqrt(np.dot(Mhyd.Kdens_m, np.exp(Mhyd.samples.T)) / cf_prof * Mhyd.transf)

    grad_dens = np.dot(Mhyd.Kdens_grad, np.exp(Mhyd.samples.T)) / 2. / dens_m ** 2 / cf_prof * Mhyd.transf

    t3d = np.dot(Mhyd.GPop, Mhyd.samppar.T)

    rout_mul = np.repeat(rout_m, nsamp).reshape(nvalm, nsamp) * cgskpc

    grad_t3d = rout_mul / cgskpc / t3d * np.dot(Mhyd.GPgrad, Mhyd.samppar.T)

    mass = - rout_mul * t3d / (cgsG * cgsamu * Mhyd.mup) * (
                grad_t3d + grad_dens) * kev2erg

    vol = 4. / 3. * np.pi * rout_m ** 3 * cgskpc ** 3

    rhoc = Mhyd.cosmo.critical_density(Mhyd.redshift).value

    delta_prof = mass.T / vol / rhoc

    for i in range(nsamp):

        ti = i * thin

        temp_func = lambda x: (np.interp(x, rout_m, delta_prof[ti, :]) - delta) ** 2

        res = minimize(temp_func, r0, method='Nelder-Mead')

        rdelta[i] = res['x'][0]

        mdelta[i] = 4. / 3. * np.pi * rdelta[i] ** 3 * cgskpc ** 3 * delta * Mhyd.cosmo.critical_density(
            Mhyd.redshift).value / Msun

        mgdelta[i] = mgas_delta(rdelta[i], Mhyd.samples[ti], Mhyd, fit_bkg=Mhyd.fit_bkg, rout_m=rout_m)

        fgdelta[i] = mgdelta[i] / mdelta[i]

    rd, rdlo, rdhi = np.percentile(rdelta, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.])

    md, mdlo, mdhi = np.percentile(mdelta, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.])

    mgd, mgdlo, mgdhi = np.percentile(mgdelta, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.])

    fgd, fgdlo, fgdhi = np.percentile(fgdelta, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.])

    dict = {
        "R_DELTA": rd,
        "R_DELTA_LO": rdlo,
        "R_DELTA_HI": rdhi,
        "M_DELTA": md,
        "M_DELTA_LO": mdlo,
        "M_DELTA_HI": mdhi,
        "MGAS_DELTA": mgd,
        "MGAS_DELTA_LO": mgdlo,
        "MGAS_DELTA_HI": mgdhi,
        "FGAS_DELTA": fgd,
        "FGAS_DELTA_LO": fgdlo,
        "FGAS_DELTA_HI": fgdhi
    }

    if plot:

        plt.clf()

        fig = plt.figure(figsize=(13, 10))

        ax_size = [0.14, 0.12,
                   0.85, 0.85]

        ax = fig.add_axes(ax_size)

        ax.minorticks_on()

        ax.tick_params(length=20, width=1, which='major', direction='in', right=True, top=True)

        ax.tick_params(length=10, width=1, which='minor', direction='in', right=True, top=True)

        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(22)

        plt.hist(mdelta, bins=30, density=True)

        plt.xlabel('$M_{\Delta} [M_\odot]$', fontsize=40)

        plt.ylabel('Frequency', fontsize=40)

        return dict, fig

    else:

        return dict


from scipy.optimize import minimize


def calc_rdelta_mdelta_forward(delta, Mhyd, Forward, plot=False, r0=500., rmax=4000., thin=10):
    '''
    For a given input overdensity Delta, compute R_delta, M_delta, Mgas_delta, fgas_delta and their uncertainties from a loaded Forward mass reconstruction

    :param delta: Overdensity with respect to critical
    :type delta: float
    :param Mhyd: :class:`hydromass.mhyd.Mhyd` object containing the results of mass reconstruction run
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param Forward: :class:`hydromass.forward.Forward` model
    :type Forward: class:`hydromass.forward.Forward`
    :param plot: If plot=True, returns a matplotlib.pyplot.figure drawing the mass distribution of the chains at R_delta. In case plot=False the function returns an empty figure.
    :type plot: bool
    :param r0: Initial value to initiate the search for the overdensity radius (in kpc). Defaults to 500
    :type r0: float
    :param rmax: Maximum radius where to search for the overdensity radius (in kpc). Defaults to 4000
    :type rmax: float
    :param thin: Thinning factor for the calculation. If the number of samples is large the calculation can be long. If thin is greater than 1 then the calculation while be done for fewer samples every "thin" value. Defaults to 10.
    :type thin: int
    :return:  Dictionary containing values R_delta, M_delta, Mgas_delta, Fgas_delta and their 1-sigma percentiles, figure if plot=True
    :rtype:
        - dict{12xfloat}
        - matplotlib.pyplot.figure
    '''

    rout = np.logspace(np.log10(Mhyd.sbprof.bins[0] * Mhyd.amin2kpc), np.log10(rmax), 100)

    rin = np.roll(rout, 1)

    rin[0] = 0.

    rref = (rin + rout) / 2.

    dr = (rout - rin) / Mhyd.amin2kpc

    area = 2. * np.pi * rref / Mhyd.amin2kpc * dr

    try:
        nn = len(Mhyd.lumfact)

    except TypeError:

        lf = Mhyd.lumfact

    else:
        lf = np.interp(rref / Mhyd.amin2kpc, Mhyd.sbprof.bins, Mhyd.lumfact)  # luminosity conversion factor

    sourcereg = np.where(rref < rmax)

    Ksb = calc_sb_operator(rref / Mhyd.amin2kpc, sourcereg, Mhyd.pars, withbkg=Mhyd.fit_bkg)

    rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore=Mhyd.nmore)

    Kdens = calc_density_operator(rout / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc, withbkg=Mhyd.fit_bkg)

    if Mhyd.cf_prof is not None:

        rref = (rin + rout) / 2.

        rad = Mhyd.sbprof.bins

        cfp = np.interp(rref, rad * Mhyd.amin2kpc, Mhyd.ccf)

    else:

        cfp = Mhyd.ccf

    nsamp = int(len(Mhyd.samppar) / thin)

    mdelta, rdelta, mgdelta, fgdelta = np.empty(nsamp), np.empty(nsamp), np.empty(nsamp), np.empty(nsamp)

    lxdelta, ktdelta = np.empty(nsamp), np.empty(nsamp)

    rhoc = Mhyd.cosmo.critical_density(Mhyd.redshift).value

    for i in range(nsamp):

        ti = i * thin

        tpar = np.array([Mhyd.samppar[ti]])

        coefs = Mhyd.samples[ti]

        dens = np.sqrt(np.dot(Kdens, np.exp(coefs)) / cfp * Mhyd.transf)

        sb = np.dot(Ksb, np.exp(coefs))

        weights = area * sb

        lumin = weights * lf

        cslum = np.cumsum(lumin)

        def temp_func(x):
            p3d = Forward.func_np(x, tpar)

            der_lnP = Forward.func_der(x, tpar)

            tne = np.interp(x, rout, dens)

            mass = - der_lnP * x * cgskpc / (
                        tne * cgsG * cgsamu * Mhyd.mup) * p3d * kev2erg

            vol = 4. / 3. * np.pi * x ** 3 * cgskpc ** 3

            delta_val = mass / vol / rhoc

            return (delta_val - delta) ** 2

        res = minimize(temp_func, r0, method='Nelder-Mead')

        rdelta[i] = res['x'][0]

        mdelta[i] = 4. / 3. * np.pi * rdelta[i] ** 3 * cgskpc ** 3 * delta * rhoc / Msun

        mgdelta[i] = mgas_delta(rdelta[i], Mhyd.samples[ti], Mhyd, fit_bkg=Mhyd.fit_bkg, rout_m=rout_m)

        fgdelta[i] = mgdelta[i] / mdelta[i]

        lxdelta[i] = np.interp(rdelta[i], rref, cslum)

        p3d = Forward.func_np(rout, tpar)

        reg = np.where(rref <= rdelta[i])

        vx = MyDeprojVol(rin[reg] / Mhyd.amin2kpc, rout[reg] / Mhyd.amin2kpc)

        vol_x = vx.deproj_vol().T

        t3d = p3d.flatten()[reg] / dens[reg]

        ei = dens[reg] ** 2 * t3d[reg] ** (-0.75)

        flux = np.dot(vol_x, ei)

        tproj = np.dot(vol_x, t3d * ei) / flux

        ktdelta[i] = np.average(tproj, weights=weights[reg])

    rd, rdlo, rdhi = np.percentile(rdelta, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.])

    md, mdlo, mdhi = np.percentile(mdelta, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.])

    mgd, mgdlo, mgdhi = np.percentile(mgdelta, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.])

    fgd, fgdlo, fgdhi = np.percentile(fgdelta, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.])

    lxd, lxdlo, lxdhi = np.percentile(lxdelta, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.])

    ktd, ktdlo, ktdhi = np.percentile(ktdelta, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.])

    yxdelta = mgdelta * ktdelta

    yxd, yxdlo, yxdhi = np.percentile(yxdelta, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.])

    all_vals = np.empty((7, nsamp))
    all_vals[0] = mdelta
    all_vals[1] = rdelta
    all_vals[2] = mgdelta
    all_vals[3] = lxdelta
    all_vals[4] = ktdelta
    all_vals[5] = yxdelta
    all_vals[6] = fgdelta

    covmat = np.corrcoef(all_vals)

    dict = {
        "R_DELTA": rd,
        "R_DELTA_LO": rdlo,
        "R_DELTA_HI": rdhi,
        "M_DELTA": md,
        "M_DELTA_LO": mdlo,
        "M_DELTA_HI": mdhi,
        "MGAS_DELTA": mgd,
        "MGAS_DELTA_LO": mgdlo,
        "MGAS_DELTA_HI": mgdhi,
        "FGAS_DELTA": fgd,
        "FGAS_DELTA_LO": fgdlo,
        "FGAS_DELTA_HI": fgdhi,
        "LX_DELTA": lxd,
        "LX_DELTA_LO": lxdlo,
        "LX_DELTA_HI": lxdhi,
        "KT_DELTA": ktd,
        "KT_DELTA_LO": ktdlo,
        "KT_DELTA_HI": ktdhi,
        "YX_DELTA": yxd,
        "YX_DELTA_LO": yxdlo,
        "YX_DELTA_HI": yxdhi
    }

    if plot:

        plt.clf()

        fig = plt.figure(figsize=(13, 10))

        ax_size = [0.14, 0.12,
                   0.85, 0.85]

        ax = fig.add_axes(ax_size)

        ax.minorticks_on()

        ax.tick_params(length=20, width=1, which='major', direction='in', right=True, top=True)

        ax.tick_params(length=10, width=1, which='minor', direction='in', right=True, top=True)

        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(22)

        plt.hist(mdelta, bins=30, density=True)

        plt.xlabel('$M_{\Delta} [M_\odot]$', fontsize=40)

        plt.ylabel('Frequency', fontsize=40)

        return dict, covmat, fig

    else:

        return dict, covmat


def calc_rdelta_mdelta_polytropic(delta, Mhyd, Polytropic, plot=False, r0=500., rmax=4000., thin=10):
    '''
    For a given input overdensity Delta, compute R_delta, M_delta, Mgas_delta, fgas_delta and their uncertainties from a loaded Forward mass reconstruction

    :param delta: Overdensity with respect to critical
    :type delta: float
    :param Mhyd: :class:`hydromass.mhyd.Mhyd` object containing the results of mass reconstruction run
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param Polytropic: :class:`hydromass.polytropic.Polytropic` model
    :type Polytropic: class:`hydromass.polytropic.Polytropic`
    :param plot: If plot=True, returns a matplotlib.pyplot.figure drawing the mass distribution of the chains at R_delta. In case plot=False the function returns an empty figure.
    :type plot: bool
    :param r0: Initial value to initiate the search for the overdensity radius (in kpc). Defaults to 500
    :type r0: float
    :param rmax: Maximum radius where to search for the overdensity radius (in kpc). Defaults to 4000
    :type rmax: float
    :param thin: Thinning factor for the calculation. If the number of samples is large the calculation can be long. If thin is greater than 1 then the calculation while be done for fewer samples every "thin" value. Defaults to 10.
    :type thin: int
    :return:  Dictionary containing values R_delta, M_delta, Mgas_delta, Fgas_delta and their 1-sigma percentiles, figure if plot=True
    :rtype:
        - dict{12xfloat}
        - matplotlib.pyplot.figure
    '''

    rout = np.logspace(np.log10(Mhyd.sbprof.bins[0] * Mhyd.amin2kpc), np.log10(rmax), 100)

    rin = np.roll(rout, 1)

    rin[0] = 0.

    rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore=Mhyd.nmore)

    Kdens = calc_density_operator(rout / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc, withbkg=Mhyd.fit_bkg)

    Kdens_grad = calc_grad_operator(rout / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc, withbkg=Mhyd.fit_bkg)

    if Mhyd.cf_prof is not None:

        rref = (rin + rout) / 2.

        rad = Mhyd.sbprof.bins

        cfp = np.interp(rref, rad * Mhyd.amin2kpc, Mhyd.ccf)

    else:

        cfp = Mhyd.ccf

    nsamp = int(len(Mhyd.samppar) / thin)

    mdelta, rdelta, mgdelta, fgdelta = np.empty(nsamp), np.empty(nsamp), np.empty(nsamp), np.empty(nsamp)

    rhoc = Mhyd.cosmo.critical_density(Mhyd.redshift).value

    for i in range(nsamp):

        ti = i * thin

        def temp_func(x):

            coefs = Mhyd.samples[ti]

            dens = np.sqrt(np.dot(Kdens, np.exp(coefs)) / cfp * Mhyd.transf)

            tpar = np.array([Mhyd.samppar[ti]])

            grad_dens = np.dot(Kdens_grad, np.exp(coefs)) / 2. / dens ** 2 / cfp * Mhyd.transf

            tne = np.interp(x, rout, dens)

            tgrad = np.interp(x, rout, grad_dens)

            p3d = Polytropic.func_np(x, tpar, tne, tgrad)

            der_lnP = Polytropic.func_der(x, tpar, dens, tgrad)

            mass = - der_lnP * x * cgskpc / (tne * cgsG * cgsamu * Mhyd.mup) * p3d * kev2erg

            vol = 4. / 3. * np.pi * x ** 3 * cgskpc ** 3

            delta_val = mass / vol / rhoc

            return (delta_val - delta) ** 2

        res = minimize(temp_func, r0, method='Nelder-Mead')

        rdelta[i] = res['x'][0]

        mdelta[i] = 4. / 3. * np.pi * rdelta[i] ** 3 * cgskpc ** 3 * delta * rhoc / Msun

        mgdelta[i] = mgas_delta(rdelta[i], Mhyd.samples[ti], Mhyd, fit_bkg=Mhyd.fit_bkg, rout_m=rout_m)

        fgdelta[i] = mgdelta[i] / mdelta[i]

    rd, rdlo, rdhi = np.percentile(rdelta, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.])

    md, mdlo, mdhi = np.percentile(mdelta, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.])

    mgd, mgdlo, mgdhi = np.percentile(mgdelta, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.])

    fgd, fgdlo, fgdhi = np.percentile(fgdelta, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.])

    dict = {
        "R_DELTA": rd,
        "R_DELTA_LO": rdlo,
        "R_DELTA_HI": rdhi,
        "M_DELTA": md,
        "M_DELTA_LO": mdlo,
        "M_DELTA_HI": mdhi,
        "MGAS_DELTA": mgd,
        "MGAS_DELTA_LO": mgdlo,
        "MGAS_DELTA_HI": mgdhi,
        "FGAS_DELTA": fgd,
        "FGAS_DELTA_LO": fgdlo,
        "FGAS_DELTA_HI": fgdhi
    }

    if plot:

        plt.clf()

        fig = plt.figure(figsize=(13, 10))

        ax_size = [0.14, 0.12,
                   0.85, 0.85]

        ax = fig.add_axes(ax_size)

        ax.minorticks_on()

        ax.tick_params(length=20, width=1, which='major', direction='in', right=True, top=True)

        ax.tick_params(length=10, width=1, which='minor', direction='in', right=True, top=True)

        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(22)

        plt.hist(mdelta, bins=30, density=True)

        plt.xlabel('$M_{\Delta} [M_\odot]$', fontsize=40)

        plt.ylabel('Frequency', fontsize=40)

        return dict, fig

    else:

        return dict



def write_all_mdelta(Mhyd, model, outfile=None, r0=500., rmax=4000., thin=10):
    """
    Write the results of the mass reconstruction run evaluated at overdensities 2500, 1000, 500, and 200 to an output file.

    :param Mhyd: Mhyd object containing the result of mass reconstruction run
    :type Mhyd: class Mhyd
    :param model: Model object defining the mass model
    :type model: class Model
    :param outfile: Name of output file. In case outfile=None (default), the function writes to Mhyd.dir/'name'.jou , with 'name' the name of the mass model.
    :type outfile: str
    """

    if outfile is None:

        outfile = Mhyd.dir + '/' + model.massmod + '.jou'

    deltafit = model.delta

    fout = open(outfile, 'w')

    fout.write('Delta fit: %g\n' %(model.delta))

    for i in range(model.npar):

        medpar, parlo, parhi = np.percentile(Mhyd.samppar[:, i], [50., 50. - 68.3 / 2., 50. + 68.3 / 2.])

        fout.write('%s    %g  (%g , %g)\n' % (model.parnames[i], medpar, parlo, parhi) )

    medp0, p0l, p0h = np.percentile(np.exp(Mhyd.samplogp0), [50., 50. - 68.3 / 2., 50. + 68.3 / 2.])

    fout.write('p0  %.3e (%.3e , %.3e)\n' % (medp0, p0l, p0h) )

    fout.write("Delta  M_delta                                 R_delta            Mgas                                   fgas\n")

    delta_vals = [2500, 1000, 500, 200]

    for delta in delta_vals:

        res = calc_rdelta_mdelta(delta, Mhyd, model, r0=r0, rmax=rmax, thin=thin)

        fout.write("%4.0f   %.4E (%.4E , %.4E)    %.0f (%.0f , %.0f)    %.4E (%.4E , %.4E)   %.4f (%.4f , %.4f)\n" % (
        delta,  res['M_DELTA'], res['M_DELTA_LO'], res['M_DELTA_HI'], res['R_DELTA'], res['R_DELTA_LO'], res['R_DELTA_HI'],
        res['MGAS_DELTA'], res['MGAS_DELTA_LO'], res['MGAS_DELTA_HI'], res['FGAS_DELTA'], res['FGAS_DELTA_LO'], res['FGAS_DELTA_HI']) )

    fout.close()

def write_all_mdelta_GP(Mhyd, outfile=None, r0=500., thin=10):
    """
    Write the results of the mass reconstruction run evaluated at overdensities 2500, 1000, 500, and 200 to an output file. In case the fitted model is noisy and shows local (or non-local) reversals, the procedure can fail if the function Delta(r)-Delta does not change sign over the range of interest. In this case, consider changing the values of rmin and rmax

    :param Mhyd: Mhyd object containing the result of mass reconstruction run
    :type Mhyd: class Mhyd
    :param outfile: Name of output file. In case outfile=None (default), the function writes to Mhyd.dir/'name'.jou , with 'name' the name of the mass model.
    :type outfile: str
    :param r0: Initial value to initiate the search for the overdensity radius (in kpc). Defaults to 500
    :type r0: float
    """

    if outfile is None:

        outfile = Mhyd.dir + '/GP.jou'

    if Mhyd.GPop is None:

        print('No GP reconstruction found in structure, skipping')

        return

    fout = open(outfile, 'w')

    fout.write("Delta  M_delta                                 R_delta            Mgas                                   fgas\n")

    delta_vals = [2500, 1000, 500, 200]

    for delta in delta_vals:

        res = calc_rdelta_mdelta_GP(delta, Mhyd, r0=r0, thin=thin)

        fout.write("%4.0f   %.4E (%.4E , %.4E)    %.0f (%.0f , %.0f)    %.4E (%.4E , %.4E)   %.4f (%.4f , %.4f)\n" % (
        delta,  res['M_DELTA'], res['M_DELTA_LO'], res['M_DELTA_HI'], res['R_DELTA'], res['R_DELTA_LO'], res['R_DELTA_HI'],
        res['MGAS_DELTA'], res['MGAS_DELTA_LO'], res['MGAS_DELTA_HI'], res['FGAS_DELTA'], res['FGAS_DELTA_LO'], res['FGAS_DELTA_HI']) )

    fout.close()

def write_covmat(covmat, outfile, res, delta):

    hdus = fits.HDUList(hdus=[fits.PrimaryHDU()])

    col1 = fits.Column(name='M_DELTA', format='E', array=covmat[:,0])
    col2 = fits.Column(name='R_DELTA', format='E', array=covmat[:,1])
    col3 = fits.Column(name='MGAS', format='E', array=covmat[:,2])
    col4 = fits.Column(name='LX', format='E', array=covmat[:,3])
    col5 = fits.Column(name='KT', format='E', array=covmat[:,4])
    col6 = fits.Column(name='YX', format='E', array=covmat[:,5])
    col7 = fits.Column(name='FGAS', format='E', array=covmat[:,6])

    cols = [col1, col2, col3, col4, col5, col6, col7]

    coldefs = fits.ColDefs(cols)

    modhdu = fits.BinTableHDU.from_columns(coldefs)

    modhdu.name = 'COVMAT'

    headhdu = modhdu.header

    headhdu['DELTA'] = delta

    for key, value in res.items():

        headhdu[key] = value

    hdus.append(modhdu)

    hdus.writeto(outfile, overwrite=True)


def write_all_mdelta_forward(Mhyd, Forward, outfile=None, r0=500., thin=10):
    """
    Write the results of the mass reconstruction run evaluated at overdensities 2500, 1000, 500, and 200 to an output file. In case the fitted model is noisy and shows local (or non-local) reversals, the procedure can fail if the function Delta(r)-Delta does not change sign over the range of interest. In this case, consider changing the values of rmin and rmax

    :param Mhyd: Mhyd object containing the result of mass reconstruction run
    :type Mhyd: class Mhyd
    :param outfile: Name of output file. In case outfile=None (default), the function writes to Mhyd.dir/'name'.jou , with 'name' the name of the mass model.
    :type outfile: str
    :param r0: Initial value to initiate the search for the overdensity radius (in kpc). Defaults to 500
    :type r0: float
    """

    if outfile is None:

        outfile = Mhyd.dir + '/FORW.jou'

    fout = open(outfile, 'w')

    fout.write("Delta  M_delta                                 R_delta            Mgas                                   fgas"
               "                       Lx                                     Tx                         Yx\n")

    delta_vals = [2500, 1000, 500, 200]

    for delta in delta_vals:

        res, covmat = calc_rdelta_mdelta_forward(delta, Mhyd, Forward, r0=r0, thin=thin)

        fout.write("%4.0f   %.4E (%.4E , %.4E)    %.0f (%.0f , %.0f)    %.4E (%.4E , %.4E)   %.4f (%.4f , %.4f)   %.4E (%.4E , %.4E)   %.4f (%.4f , %.4f)   %.4E (%.4E , %.4E)\n" % (
        delta,  res['M_DELTA'], res['M_DELTA_LO'], res['M_DELTA_HI'], res['R_DELTA'], res['R_DELTA_LO'], res['R_DELTA_HI'],
        res['MGAS_DELTA'], res['MGAS_DELTA_LO'], res['MGAS_DELTA_HI'], res['FGAS_DELTA'], res['FGAS_DELTA_LO'], res['FGAS_DELTA_HI'],
        res['LX_DELTA'], res['LX_DELTA_LO'], res['LX_DELTA_HI'], res['KT_DELTA'], res['KT_DELTA_LO'], res['KT_DELTA_HI'],
        res['YX_DELTA'], res['YX_DELTA_LO'], res['YX_DELTA_HI']))

        outcov = '%s/covmat_%d.fits' % (Mhyd.dir, delta)

        write_covmat(covmat, outcov, res, delta)

    fout.close()
