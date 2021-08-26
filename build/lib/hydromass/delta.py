from .constants import *
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from .deproject import calc_density_operator
from .plots import rads_more

def delta_func(r, Mhyd, model, pars):
    """
    Return profile of overdensity Delta with respect to critical density for a given input mass model

    :param r: Radii in kpc
    :type r: numpy.ndarray or float
    :param Mhyd: Mhyd object containing mass reconstruction
    :type Mhyd: class Mhyd
    :param model:Model object containing the definition of the mass model
    :type model: class Model
    :param pars:Parameter vector to be passed to the mass model
    :type pars: numpy.ndarray
    :return: Overdensity as a function of radius
    :rtype: numpy.ndarray
    """

    mass = Mhyd.mfact * model.func_np(r, pars, delta=model.delta) * 1e13 * Msun

    vol = 4. / 3. * np.pi * r ** 3 * cgskpc ** 3

    rhoc = Mhyd.cosmo.critical_density(Mhyd.redshift).value

    return mass / vol / rhoc


def delta_func_GP(r, Mhyd, mass):
    """
    Return profile of overdensity Delta with respect to critical density for a given input mass model

    :param r: Radii in kpc
    :type r: numpy.ndarray or float
    :param Mhyd: Mhyd object containing mass reconstruction
    :type Mhyd: class Mhyd
    :param model:Model object containing the definition of the mass model
    :type model: class Model
    :param pars:Parameter vector to be passed to the mass model
    :type pars: numpy.ndarray
    :return: Overdensity as a function of radius
    :rtype: numpy.ndarray
    """
    vol = 4. / 3. * np.pi * r ** 3 * cgskpc ** 3

    rhoc = Mhyd.cosmo.critical_density(Mhyd.redshift).value

    return mass / vol / rhoc


def mgas_delta(rdelta, coefs, Mhyd, fit_bkg=False, rout_m=None):
    """
    Compute Mgas at an input R_delta

    :param rdelta: R_delta in kpc
    :type rdelta: float
    :param coefs: Coefficients describing the density profile
    :type coefs: numpy.ndarray
    :param Mhyd: Mhyd object containing reconstruction
    :type Mhyd: class Mhyd
    :return: Gas mass evaluated exactly at rdelta
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
    :param Mhyd: Mhyd object containing reconstruction
    :type Mhyd: class Mhyd
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


def calc_rdelta_mdelta(delta, Mhyd, model, plot=False, rmin=100., rmax=4000.):
    """
    For a given input overdensity Delta, compute R_delta, M_delta, Mgas_delta, fgas_delta and their uncertainties

    :param delta: Overdensity with respect to critical
    :type float
    :param Mhyd: Mhyd object containing the results of mass reconstruction run
    :type Mhyd: class Mhyd
    :param model: Model object defining the mass model
    :type model: class Model
    :param plot: If plot=True, returns a matplotlib.pyplot.figure drawing the mass distribution of the chains at R_delta. In case plot=False the function returns an empty figure.
    :type plot: bool
    :return:  Dictionary containing values R_delta, M_delta, Mgas_delta, Fgas_delta and their 1-sigma percentiles, figure if plot=True
    :rtype  dict{12xfloat}, class matplotlib.pyplot.figure
    """

    nsamp = len(Mhyd.samppar)

    mdelta, rdelta, mgdelta, fgdelta = np.empty(nsamp), np.empty(nsamp), np.empty(nsamp), np.empty(nsamp)

    rin_m, rout_m, index_x, index_sz, sum_mat = rads_more(Mhyd, nmore=Mhyd.nmore)

    for i in range(nsamp):

        if Mhyd.dmonly:

            r_mbar, mbar_ov = mbar_overdens(rmax, Mhyd.samples[i], Mhyd, fit_bkg = Mhyd.fit_bkg, rout_m=rout_m)

            temp_func = lambda x: delta_func(np.array([x]), Mhyd, model, np.array([Mhyd.samppar[i]])) + np.interp(x, r_mbar, mbar_ov) - delta

        else:

            temp_func = lambda x: delta_func(np.array([x]), Mhyd, model, np.array([Mhyd.samppar[i]])) - delta

        rdelta[i] = brentq(temp_func, rmin, rmax)

        mdelta[i] = 4. / 3. * np.pi * rdelta[i] ** 3 * cgskpc ** 3 * delta * Mhyd.cosmo.critical_density(Mhyd.redshift).value / Msun

        mgdelta[i] = mgas_delta(rdelta[i], Mhyd.samples[i], Mhyd, fit_bkg = Mhyd.fit_bkg, rout_m=rout_m)

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

        return  dict, fig

    else:

        return dict


from scipy.optimize import brentq


def calc_rdelta_mdelta_GP(delta, Mhyd, plot=False, rmin=100., rmax=4000.):
    """
    For a given input overdensity Delta, compute R_delta, M_delta, Mgas_delta, fgas_delta and their uncertainties

    :param delta: Overdensity with respect to critical
    :type float
    :param Mhyd: Mhyd object containing the results of mass reconstruction run
    :type Mhyd: class:`hydromass.Mhyd`
    :param plot: If plot=True, returns a matplotlib.pyplot.figure drawing the mass distribution of the chains at R_delta. In case plot=False the function returns an empty figure.
    :type plot: bool
    :return:  Dictionary containing values R_delta, M_delta, Mgas_delta, Fgas_delta and their 1-sigma percentiles, figure if plot=True
    :rtype  dict{12xfloat}, class matplotlib.pyplot.figure
    """

    nsamp = len(Mhyd.samppar)

    mdelta, rdelta, mgdelta, fgdelta = np.empty(nsamp), np.empty(nsamp), np.empty(nsamp), np.empty(nsamp)

    rin_m, rout_m, index_x, index_sz, sum_mat = rads_more(Mhyd, nmore=Mhyd.nmore)

    nvalm = len(rin_m)

    if Mhyd.cf_prof is not None:

        cf_prof = np.repeat(Mhyd.cf_prof, nsamp).reshape(nvalm, nsamp)

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
        temp_func = lambda x: np.interp(x, rout_m, delta_prof[i, :]) - delta

        rdelta[i] = brentq(temp_func, rmin, rmax)

        mdelta[i] = 4. / 3. * np.pi * rdelta[i] ** 3 * cgskpc ** 3 * delta * Mhyd.cosmo.critical_density(
            Mhyd.redshift).value / Msun

        mgdelta[i] = mgas_delta(rdelta[i], Mhyd.samples[i], Mhyd, fit_bkg=Mhyd.fit_bkg, rout_m=rout_m)

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

def calc_rdelta_mdelta_forward(delta, Mhyd, Forward, plot=False, rmin=100., rmax=4000.):
    """
    For a given input overdensity Delta, compute R_delta, M_delta, Mgas_delta, fgas_delta and their uncertainties

    :param delta: Overdensity with respect to critical
    :type float
    :param Mhyd: Mhyd object containing the results of mass reconstruction run
    :type Mhyd: class:`hydromass.Mhyd`
    :param plot: If plot=True, returns a matplotlib.pyplot.figure drawing the mass distribution of the chains at R_delta. In case plot=False the function returns an empty figure.
    :type plot: bool
    :return:  Dictionary containing values R_delta, M_delta, Mgas_delta, Fgas_delta and their 1-sigma percentiles, figure if plot=True
    :rtype  dict{12xfloat}, class matplotlib.pyplot.figure
    """

    nsamp = len(Mhyd.samppar)

    mdelta, rdelta, mgdelta, fgdelta = np.empty(nsamp), np.empty(nsamp), np.empty(nsamp), np.empty(nsamp)

    rin_m, rout_m, index_x, index_sz, sum_mat = rads_more(Mhyd, nmore=Mhyd.nmore)

    nvalm = len(rin_m)

    if Mhyd.cf_prof is not None:

        cf_prof = np.repeat(Mhyd.cf_prof, nsamp).reshape(nvalm, nsamp)

    else:

        cf_prof = Mhyd.ccf

    dens_m = np.sqrt(np.dot(Mhyd.Kdens_m, np.exp(Mhyd.samples.T)) / cf_prof * Mhyd.transf)

    p3d = Forward.func_np(rout_m, Mhyd.samppar)

    der_lnP = Forward.func_der(rout_m, Mhyd.samppar)

    rout_mul = np.repeat(rout_m, nsamp).reshape(nvalm, nsamp) * cgskpc

    mass = - der_lnP * rout_mul / (dens_m * cgsG * cgsamu * Mhyd.mup) * p3d * kev2erg

    vol = 4. / 3. * np.pi * rout_m ** 3 * cgskpc ** 3

    rhoc = Mhyd.cosmo.critical_density(Mhyd.redshift).value

    delta_prof = mass.T / vol / rhoc

    for i in range(nsamp):
        temp_func = lambda x: np.interp(x, rout_m, delta_prof[i, :]) - delta

        rdelta[i] = brentq(temp_func, rmin, rmax)

        mdelta[i] = 4. / 3. * np.pi * rdelta[i] ** 3 * cgskpc ** 3 * delta * Mhyd.cosmo.critical_density(
            Mhyd.redshift).value / Msun

        mgdelta[i] = mgas_delta(rdelta[i], Mhyd.samples[i], Mhyd, fit_bkg=Mhyd.fit_bkg, rout_m=rout_m)

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

def calc_rdelta_mdelta_polytropic(delta, Mhyd, Polytropic, plot=False, rmin=100., rmax=4000.):
    """
    For a given input overdensity Delta, compute R_delta, M_delta, Mgas_delta, fgas_delta and their uncertainties

    :param delta: Overdensity with respect to critical
    :type float
    :param Mhyd: Mhyd object containing the results of mass reconstruction run
    :type Mhyd: class:`hydromass.Mhyd`
    :param plot: If plot=True, returns a matplotlib.pyplot.figure drawing the mass distribution of the chains at R_delta. In case plot=False the function returns an empty figure.
    :type plot: bool
    :return:  Dictionary containing values R_delta, M_delta, Mgas_delta, Fgas_delta and their 1-sigma percentiles, figure if plot=True
    :rtype  dict{12xfloat}, class matplotlib.pyplot.figure
    """

    nsamp = len(Mhyd.samppar)

    mdelta, rdelta, mgdelta, fgdelta = np.empty(nsamp), np.empty(nsamp), np.empty(nsamp), np.empty(nsamp)

    rin_m, rout_m, index_x, index_sz, sum_mat = rads_more(Mhyd, nmore=Mhyd.nmore)

    nvalm = len(rin_m)

    if Mhyd.cf_prof is not None:

        cf_prof = np.repeat(Mhyd.cf_prof, nsamp).reshape(nvalm, nsamp)

    else:

        cf_prof = Mhyd.ccf

    dens_m = np.sqrt(np.dot(Mhyd.Kdens_m, np.exp(Mhyd.samples.T)) / Mhyd.ccf * Mhyd.transf)

    grad_dens = np.dot(Mhyd.Kdens_grad, np.exp(Mhyd.samples.T)) / 2. / dens_m ** 2 / Mhyd.ccf * Mhyd.transf

    p3d = Polytropic.func_np(rout_m, Mhyd.samppar, dens_m, grad_dens)

    der_lnP = Polytropic.func_der(rout_m, Mhyd.samppar, dens_m, grad_dens)

    rout_mul = np.repeat(rout_m, nsamp).reshape(nvalm, nsamp) * cgskpc

    mass = - der_lnP * rout_mul / (dens_m * cgsG * cgsamu * Mhyd.mup) * p3d * kev2erg

    vol = 4. / 3. * np.pi * rout_m ** 3 * cgskpc ** 3

    rhoc = Mhyd.cosmo.critical_density(Mhyd.redshift).value

    delta_prof = mass.T / vol / rhoc

    for i in range(nsamp):
        temp_func = lambda x: np.interp(x, rout_m, delta_prof[i, :]) - delta

        rdelta[i] = brentq(temp_func, rmin, rmax)

        mdelta[i] = 4. / 3. * np.pi * rdelta[i] ** 3 * cgskpc ** 3 * delta * Mhyd.cosmo.critical_density(
            Mhyd.redshift).value / Msun

        mgdelta[i] = mgas_delta(rdelta[i], Mhyd.samples[i], Mhyd, fit_bkg=Mhyd.fit_bkg, rout_m=rout_m)

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



def write_all_mdelta(Mhyd, model, outfile=None, rmin=100., rmax=2000.):
    """
    Write the results of the mass reconstruction run evaluated at overdensities 2500, 1000, 500, and 200 to an output file.

    :param Mhyd: Mhyd object containing the result of mass reconstruction run
    :type Mhyd: class Mhyd
    :param model: Model object defining the mass model
    :type model: class Model
    :param outfile: Name of output file. In case outfile=None (default), the function writes to Mhyd.dir/'name'.jou , with 'name' the name of the mass model.
    :type outfile: str
    :return: None
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

        res = calc_rdelta_mdelta(delta, Mhyd, model, rmin=rmin, rmax=rmax)

        fout.write("%4.0f   %.4E (%.4E , %.4E)    %.0f (%.0f , %.0f)    %.4E (%.4E , %.4E)   %.4f (%.4f , %.4f)\n" % (
        delta,  res['M_DELTA'], res['M_DELTA_LO'], res['M_DELTA_HI'], res['R_DELTA'], res['R_DELTA_LO'], res['R_DELTA_HI'],
        res['MGAS_DELTA'], res['MGAS_DELTA_LO'], res['MGAS_DELTA_HI'], res['FGAS_DELTA'], res['FGAS_DELTA_LO'], res['FGAS_DELTA_HI']) )

    fout.close()

def write_all_mdelta_GP(Mhyd, outfile=None, rmin=100., rmax=2000.):
    """
    Write the results of the mass reconstruction run evaluated at overdensities 2500, 1000, 500, and 200 to an output file.

    :param Mhyd: Mhyd object containing the result of mass reconstruction run
    :type Mhyd: class Mhyd
    :param outfile: Name of output file. In case outfile=None (default), the function writes to Mhyd.dir/'name'.jou , with 'name' the name of the mass model.
    :type outfile: str
    :return: None
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

        res = calc_rdelta_mdelta_GP(delta, Mhyd, rmin=100., rmax=2000.)

        fout.write("%4.0f   %.4E (%.4E , %.4E)    %.0f (%.0f , %.0f)    %.4E (%.4E , %.4E)   %.4f (%.4f , %.4f)\n" % (
        delta,  res['M_DELTA'], res['M_DELTA_LO'], res['M_DELTA_HI'], res['R_DELTA'], res['R_DELTA_LO'], res['R_DELTA_HI'],
        res['MGAS_DELTA'], res['MGAS_DELTA_LO'], res['MGAS_DELTA_HI'], res['FGAS_DELTA'], res['FGAS_DELTA_LO'], res['FGAS_DELTA_HI']) )

    fout.close()


def write_all_mdelta_forward(Mhyd, Forward, outfile=None, rmin=100., rmax=2000.):
    """
    Write the results of the mass reconstruction run evaluated at overdensities 2500, 1000, 500, and 200 to an output file.

    :param Mhyd: Mhyd object containing the result of mass reconstruction run
    :type Mhyd: class Mhyd
    :param outfile: Name of output file. In case outfile=None (default), the function writes to Mhyd.dir/'name'.jou , with 'name' the name of the mass model.
    :type outfile: str
    :return: None
    """

    if outfile is None:

        outfile = Mhyd.dir + '/FORW.jou'

    fout = open(outfile, 'w')

    fout.write("Delta  M_delta                                 R_delta            Mgas                                   fgas\n")

    delta_vals = [2500, 1000, 500, 200]

    for delta in delta_vals:

        res = calc_rdelta_mdelta_forward(delta, Mhyd, Forward, rmin=100., rmax=2000.)

        fout.write("%4.0f   %.4E (%.4E , %.4E)    %.0f (%.0f , %.0f)    %.4E (%.4E , %.4E)   %.4f (%.4f , %.4f)\n" % (
        delta,  res['M_DELTA'], res['M_DELTA_LO'], res['M_DELTA_HI'], res['R_DELTA'], res['R_DELTA_LO'], res['R_DELTA_HI'],
        res['MGAS_DELTA'], res['MGAS_DELTA_LO'], res['MGAS_DELTA_HI'], res['FGAS_DELTA'], res['FGAS_DELTA_LO'], res['FGAS_DELTA_HI']) )

    fout.close()