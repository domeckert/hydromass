import numpy as np
import pymc as pm
from .deproject import *
from .plots import rads_more, get_coolfunc, plt
from .constants import *

# GNFW function should work both for numpy.ndarray and pymc3/theano formats
def gnfw_pm(rad, p0, c500, gamma, alfa, beta):
    '''
    Theano function defining the generalized NFW profile

    .. math::

        P_{gNFW}(r) = \\frac{P_0} {(c_{500} r)^\\gamma (1+(c_{500} r)^\\alpha)^{(\\beta-\\gamma)/\\alpha}}

    :param rad: Radius
    :type rad: theano.tensor
    :param p0: :math:`P_0` parameter
    :type p0: theano.tensor
    :param c500: :math:`c_{500}` parameter
    :type c500: theano.tensor
    :param gamma: :math:`\\gamma` parameter
    :type gamma: theano.tensor
    :param alfa: :math:`\\alpha` parameter
    :type alfa: theano.tensor
    :param beta: :math:`\\beta` parameter
    :type beta: theano.tensor
    :return: Model pressure
    :rtype: theano.tensor
    '''

    x = c500 * rad

    t1 = x ** gamma

    t2 = (1. + x ** alfa) ** ( (beta - gamma) / alfa)

    fgnfw = p0 / t1 / t2

    return  fgnfw


def gnfw_np(xout, pars):
    '''
    Numpy function defining the generalized NFW profile

    .. math::

        P_{gNFW}(r) = \\frac{P_0} {(c_{500} r)^\\gamma (1+(c_{500} r)^\\alpha)^{(\\beta-\\gamma)/\\alpha}}

    :param rad: 1-D array with radius definition
    :type rad: numpy.ndarray
    :param pars: 2-D array including the parameter samples. Column order is: p0, c500, gamma, alpha, beta
    :type pars: numpy.ndarray
    :return: 2-D array including all the realizations of the model pressure profile
    :rtype: numpy.ndarray
    '''


    p0 = pars[:, 0]

    c500 = pars[:, 1]

    gamma = pars[:, 2]

    alfa = pars[:, 3]

    beta = pars[:, 4]

    npt = len(xout)

    npars = len(p0)

    p0mul = np.repeat(p0, npt).reshape(npars, npt)

    c500mul = np.repeat(c500, npt).reshape(npars, npt)

    gammamul = np.repeat(gamma, npt).reshape(npars, npt)

    alfamul = np.repeat(alfa, npt).reshape(npars, npt)

    betamul = np.repeat(beta, npt).reshape(npars, npt)

    xoutmul = np.tile(xout, npars).reshape(npars, npt)

    x = c500mul * xoutmul

    t1 = x ** gammamul

    t2 = (1. + x ** alfamul) ** ((betamul - gammamul) / alfamul)

    fgnfw = p0mul / t1 / t2

    return fgnfw.T

# Pressure gradient from GNFW function
def der_lnP_np(xout, pars):
    '''
    Analytic logarithmic derivative of the generalized NFW function

    .. math::

        \\frac{d \\ln P}{d \\ln r} = - \\left( \\gamma + \\frac{(\\beta - \\gamma)(c_{500}r)^{\\alpha}} {1 + (c_{500}r)^{\\alpha} } \\right)

    :param xout: 1-D array of radii
    :type xout: numpy.ndarray
    :param pars: 2-D array including the parameter samples
    :type pars: numpy.ndarray
    :return: Pressure gradient profiles for all realizations
    :rtype: numpy.ndarray
    '''
    p0 = pars[:, 0]

    c500 = pars[:, 1]

    gamma = pars[:, 2]

    alfa = pars[:, 3]

    beta = pars[:, 4]

    npt = len(xout)

    npars = len(p0)

    c500mul = np.repeat(c500, npt).reshape(npars, npt)

    gammamul = np.repeat(gamma, npt).reshape(npars, npt)

    alfamul = np.repeat(alfa, npt).reshape(npars, npt)

    betamul = np.repeat(beta, npt).reshape(npars, npt)

    xoutmul = np.tile(xout, npars).reshape(npars, npt)

    x = c500mul * xoutmul

    t1 = (betamul - gammamul) * x ** alfamul

    t2 = 1. + x ** alfamul

    fder = - (gammamul + t1 / t2)

    return  fder.T


def kt_forw_from_samples(Mhyd, Forward, nmore=5):
    """

    Compute model temperature profile from forward mass reconstruction run evaluated at reference X-ray temperature radii

    :param Mhyd: :class:`hydromass.mhyd.Mhyd` object including the reconstruction
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param Forward: :class:`hydromass.forward.Forward` object defining the forward model
    :type Forward: class:`hydromass.forward.Forward`
    :return: Dictionary including the median temperature and 1-sigma percentiles, both 3D and spectroscopic-like
    :rtype: dict(9xnpt)
    """

    if Mhyd.spec_data is None:

        print('No spectral data provided')

        return

    nsamp = len(Mhyd.samples)

    rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore=nmore)

    nvalm = len(rin_m)

    if Mhyd.cf_prof is not None:

        rref_m = (rin_m + rout_m) / 2.

        rad = Mhyd.sbprof.bins

        tcf = np.interp(rref_m, rad * Mhyd.amin2kpc, Mhyd.ccf)

        cf_prof = np.repeat(tcf, nsamp).reshape(nvalm, nsamp)

    else:

        cf_prof = Mhyd.ccf

    vx = MyDeprojVol(rin_m / Mhyd.amin2kpc, rout_m / Mhyd.amin2kpc)

    vol_x = vx.deproj_vol().T

    if Mhyd.spec_data.psfmat is not None:

        mat1 = np.dot(Mhyd.spec_data.psfmat.T, sum_mat)

        proj_mat = np.dot(mat1, vol_x)

    else:

        proj_mat = np.dot(sum_mat, vol_x)

    npx = len(Mhyd.spec_data.rref_x)

    dens_m = np.sqrt(np.dot(Mhyd.Kdens_m, np.exp(Mhyd.samples.T)) / cf_prof * Mhyd.transf)

    p3d = Forward.func_np(rout_m, Mhyd.samppar)

    t3d = p3d / dens_m

    # Mazzotta weights
    ei = dens_m ** 2 * t3d ** (-0.75)

    # Temperature projection
    flux = np.dot(proj_mat, ei)

    tproj = np.dot(proj_mat, t3d * ei) / flux

    tmed, tlo, thi = np.percentile(tproj, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    t3dot, t3dlt, t3dht = np.percentile(t3d, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    t3do, t3dl, t3dh = t3dot[index_x], t3dlt[index_x], t3dht[index_x]

    dict = {
        "R_IN": Mhyd.spec_data.rin_x,
        "R_OUT": Mhyd.spec_data.rout_x,
        "R_REF": Mhyd.spec_data.rref_x,
        "T3D": t3do,
        "T3D_LO": t3dl,
        "T3D_HI": t3dh,
        "TSPEC": tmed,
        "TSPEC_LO": tlo,
        "TSPEC_HI": thi
    }

    return dict


def P_forw_from_samples(Mhyd, Forward, nmore=5):
    """

    Compute model pressure profile from Forward mass reconstruction run evaluated at the reference SZ radii

    :param Mhyd: :class:`hydromass.mhyd.Mhyd` object including the reconstruction
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param Forward: :class:`hydromass.forward.Forward` object defining the forward model
    :type Forward: class:`hydromass.forward.Forward`
    :return: Median pressure, Lower 1-sigma percentile, Upper 1-sigma percentile
    :rtype: float
    """

    if Mhyd.sz_data is None:

        print('No SZ data provided')

        return

    rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore=nmore)

    p3d = Forward.func_np(rout_m, Mhyd.samppar)

    pmt, plot, phit = np.percentile(p3d, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    pmed, plo, phi = pmt[index_sz], plot[index_sz], phit[index_sz]

    return pmed, plo, phi


def mass_forw_from_samples(Mhyd, Forward, plot=False, nmore=5):
    '''
    Compute the best-fit forward mass model and its 1-sigma error envelope from a loaded Forward run. 

    :param Mhyd: :class:`hydromass.mhyd.Mhyd` object including the reconstruction
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param Forward: :class:`hydromass.forward.Forward` object defining the forward model
    :type Forward: class:`hydromass.forward.Forward`
    :param plot: Produce a plot of the mass profile from the result of the forward fit. Defaults to False
    :type plot: bool
    :param nmore: Number of points defining fine grid, must be equal to the value used for the mass reconstruction. Defaults to 5
    :type nmore: int
    :return: Dictionary containing the profiles of hydrostatic mass, gas mass, and gas fraction
    :rtype: dict(11xnpt)
    '''

    nsamp = len(Mhyd.samples)

    rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore=nmore)

    nvalm = len(rin_m)

    if Mhyd.cf_prof is not None:

        rref_m = (rin_m + rout_m) / 2.

        rad = Mhyd.sbprof.bins

        tcf = np.interp(rref_m, rad * Mhyd.amin2kpc, Mhyd.ccf)

        cf_prof = np.repeat(tcf, nsamp).reshape(nvalm, nsamp)

    else:

        cf_prof = Mhyd.ccf

    dens_m = np.sqrt(np.dot(Mhyd.Kdens_m, np.exp(Mhyd.samples.T)) / cf_prof * Mhyd.transf)

    p3d = Forward.func_np(rout_m, Mhyd.samppar)

    der_lnP = Forward.func_der(rout_m, Mhyd.samppar)

    rout_mul = np.repeat(rout_m, nsamp).reshape(nvalm, nsamp) * cgskpc

    mass = - der_lnP * rout_mul / (dens_m * cgsG * cgsamu * Mhyd.mup) * p3d * kev2erg / Msun

    mmed, mlo, mhi = np.percentile(mass, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    # Matrix containing integration volumes
    volmat = np.repeat(4. / 3. * np.pi * (rout_m ** 3 - rin_m ** 3), nsamp).reshape(nvalm, nsamp)

    # Compute Mgas profile as cumulative sum over the volume

    nhconv = cgsamu * Mhyd.mu_e * cgskpc ** 3 / Msun  # Msun/kpc^3

    ones_mat = np.ones((nvalm, nvalm))

    cs_mat = np.tril(ones_mat)

    mgas = np.dot(cs_mat, dens_m * nhconv * volmat)

    mg, mgl, mgh = np.percentile(mgas, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    fgas = mgas / mass

    fg, fgl, fgh = np.percentile(fgas, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    dict = {
        "R_IN": rin_m,
        "R_OUT": rout_m,
        "MASS": mmed,
        "MASS_LO": mlo,
        "MASS_HI": mhi,
        "MGAS": mg,
        "MGAS_LO": mgl,
        "MGAS_HI": mgh,
        "FGAS": fg,
        "FGAS_LO": fgl,
        "FGAS_HI": fgh
    }

    if plot:

        fig = plt.figure(figsize=(13, 10))

        ax_size = [0.14, 0.12,
                   0.85, 0.85]

        ax = fig.add_axes(ax_size)

        ax.minorticks_on()

        ax.tick_params(length=20, width=1, which='major', direction='in', right=True, top=True)

        ax.tick_params(length=10, width=1, which='minor', direction='in', right=True, top=True)

        for item in (ax.get_xticklabels() + ax.get_yticklabels()):

            item.set_fontsize(22)

        plt.xscale('log')

        plt.yscale('log')

        plt.plot(rout_m, mg, color='blue', label='$M_{\rm gas}$')

        plt.fill_between(rout_m, mgl, mgh, color='blue', alpha=0.4)

        plt.plot(rout_m, mmed, color='red', label='$M_{\rm Hyd}$')

        plt.fill_between(rout_m, mlo, mhi, color='red', alpha=0.4)

        plt.xlabel('Radius [kpc]', fontsize=40)

        plt.ylabel('$M(<R) [M_\odot]$', fontsize=40)

        return dict, fig

    else:

        return dict

def prof_forw_hires(Mhyd, Forward, nmore=5, Z=0.3):
    """
    Compute best-fitting profiles and error envelopes from fitted data

    :param Mhyd: :class:`hydromass.mhyd.Mhyd` object including the reconstruction
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param Forward: :class:`hydromass.forward.Forward` object defining the forward model
    :type Forward: class:`hydromass.forward.Forward`
    :param nmore: Number of points defining fine grid, must be equal to the value used for the mass reconstruction. Defaults to 5
    :type nmore: int
    :param Z: Metallicity relative to Solar for the computation of the cooling function. Defaults to 0.3
    :type Z: float
    :return: Dictionary containing the profiles of thermodynamic quantities (temperature, pressure, gas density, and entropy), cooling function and cooling time
    :rtype: dict(23xnpt)
    """

    rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore=nmore)

    vx = MyDeprojVol(rin_m / Mhyd.amin2kpc, rout_m / Mhyd.amin2kpc)

    vol_x = vx.deproj_vol().T

    p3d = Forward.func_np(rout_m, Mhyd.samppar)

    nvalm = len(rin_m)

    nsamp = len(Mhyd.samples)

    if Mhyd.cf_prof is not None:

        rref_m = (rin_m + rout_m) / 2.

        rad = Mhyd.sbprof.bins

        tcf = np.interp(rref_m, rad * Mhyd.amin2kpc, Mhyd.ccf)

        cf_prof = np.repeat(tcf, nsamp).reshape(nvalm, nsamp)

    else:

        cf_prof = Mhyd.ccf

    dens_m = np.sqrt(np.dot(Mhyd.Kdens_m, np.exp(Mhyd.samples.T)) / cf_prof * Mhyd.transf)

    t3d = p3d / dens_m

    # Mazzotta weights
    ei = dens_m ** 2 * t3d ** (-0.75)

    # Temperature projection
    flux = np.dot(vol_x, ei)

    tproj = np.dot(vol_x, t3d * ei) / flux

    K3d = t3d * dens_m ** (- 2. / 3.)

    mptot, mptotl, mptoth = np.percentile(p3d, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    mt3d, mt3dl, mt3dh = np.percentile(t3d, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    mtp, mtpl, mtph = np.percentile(tproj, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    mne, mnel, mneh = np.percentile(dens_m, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    mK, mKl, mKh = np.percentile(K3d, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    coolfunc, ktgrid = get_coolfunc(Z)

    lambda3d = np.interp(t3d, ktgrid, coolfunc)

    tcool = 3./2. * dens_m * (1. + 1./Mhyd.nhc) * t3d * kev2erg / (lambda3d * dens_m **2 / Mhyd.nhc) / year

    mtc, mtcl, mtch = np.percentile(tcool, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    mcf, mcfl, mcfh = np.percentile(lambda3d, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    dict={
        "R_IN": rin_m,
        "R_OUT": rout_m,
        "P_TOT": mptot,
        "P_TOT_LO": mptotl,
        "P_TOT_HI": mptoth,
        "T3D": mt3d,
        "T3D_LO": mt3dl,
        "T3D_HI": mt3dh,
        "TSPEC": mtp,
        "TSPEC_LO": mtpl,
        "TSPEC_HI": mtph,
        "NE": mne,
        "NE_LO": mnel,
        "NE_HI": mneh,
        "K": mK,
        "K_LO": mKl,
        "K_HI": mKh,
        "T_COOL": mtc,
        "T_COOL_LO": mtcl,
        "T_COOL_HI": mtch,
        "LAMBDA": mcf,
        "LAMBDA_LO": mcfl,
        "LAMBDA_HI": mcfh
    }

    return dict



class Forward:
    """
    Class allowing the user to define a parametric forward model to the gas pressure. Currently only supports the generalized NFW model (Nagai et al. 2007), :func:`hydromass.forward.gnfw_pm`.

    :param start: 1-D array including the central values of the Gaussian priors on the gNFW model parameters. If None, the starting values are set automatically using the average gNFW model of Planck Collaboration V (2013). Defaults to None.
    :type start: numpy.ndarray
    :param sd: 1-D array including the standard deviation values of the Gaussian priors on the gNFW model parameters. If None, the standard deviations are set automatically to encompass the variety of pressure profiles of Planck Collaboration V (2013). Defaults to None.
    :type sd: numpy.ndarray
    :param limits: 2-D array including the minimum and maximum allowed values for each gNFW parameter. If None, very broad automatic boundaries are used. Defaults to None.
    :type limits: numpy.ndarray
    :param fix: 1-D array of booleans describing whether each parameter is fitted (False) or fixed to the input value given by the "start" parameter (True). If None all the parameters are fitted. Defaults to None.
    :type fix: numpy.ndarray

    """
    def __init__(self, start=None, sd=None, limits=None, fix=None):

        self.npar = 5

        self.parnames = ['p0', 'c500', 'gamma', 'alpha', 'beta']

        if start is None:
            self.start = [1e-2, 1.5/1200., 0.3, 1.3, 4.4] # similar to Universal profile from Ghirardini+19

        else:
            try:
                assert (len(start) == self.npar)
            except AssertionError:
                print('Number of starting parameters does not match function.')
                return

            self.start = start

        if sd is None:
            self.sd = [1e-2, 1.0/1200., 0.5, 0.5, 2.]

        else:

            try:
                assert (len(sd) == self.npar)
            except AssertionError:
                print('Shape of sd does not match function.')
                return

            self.sd = sd


        if limits is None:

            limits = np.empty((self.npar, 2))

            limits[0] = [1e-4, 1.]

            limits[1] = [0.1/2000., 6./500.]

            limits[2] = [0., 3.]

            limits[3] = [0.5, 2.5]

            limits[4] = [2., 10.]

        else:

            try:
                assert (limits.shape == (self.npar,2))
            except AssertionError:
                print('Shape of limits does not match function.')
                return

        if fix is None:

            self.fix = [False, False, False, False, False]

        else:

            try:
                assert (len(fix) == self.npar)
            except AssertionError:
                print('Shape of fix vectory does not match function.')
                return

            self.fix = fix

        self.limits = limits

        self.func_np = gnfw_np

        self.func_pm = gnfw_pm

        self.func_der = der_lnP_np



def Run_Forward_PyMC3(Mhyd,Forward, bkglim=None,nmcmc=1000,fit_bkg=False,back=None,
                   samplefile=None,nrc=None,nbetas=6,min_beta=0.6, nmore=5,
                   tune=500, find_map=True):
    """
    Set up parametric forward model fit and optimize with PyMC3. The routine takes a parametric function for the 3D gas pressure profile as input and optimizes jointly for the gas density and pressure profiles. The mass profile is then computed point by point using the analytic derivative of the model pressure profile:

    .. math::

        M_{forw}(<r) = - \\frac{r^2}{\\rho_{gas}(r) G} \\frac{d \\ln P}{d \\ln r}

    The gas density profile is fitted to the surface brightness profile and described as a linear combination of King functions. The definition of the parametric forward model should be defined using the :class:`hydromass.forward.Forward` class, which implements the generalized NFW model and can be used to implement any parametric model for the gas pressure. The 3D pressure profile is then projected along the line of sight an weighted by spectroscopic-like weights to predict the spectroscopic temperature profile.

    The parameters of the forward model and of the gas density profile are fitted jointly to the data. Priors on the input parameters can be set by the user in the definition of the forward model.

    :param Mhyd: A :class:`hydromass.mhyd.Mhyd` object including the loaded data and initial setup (mandatory input)
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param model:  A :class:`hydromass.forward.Forward` object including the definition of the forward model and its input values (mandatory input)
    :type model: class:`hydromass.forward.Forward`
    :param bkglim: Limit (in arcmin) out to which the SB data will be fitted; if None then the whole range is considered. Defaults to None.
    :type bkglim: float
    :param nmcmc: Number of PyMC3 steps. Defaults to 1000
    :type nmcmc: int
    :param fit_bkg: Choose whether the counts and the background will be fitted on-the-fly using a Poisson model (fit_bkg=True) or if the surface brightness will be fitted, in which case it is assumed that the background has already been subtracted and Gaussian likelihood will be used (default = False)
    :type fit_bkg: bool
    :param back: Input value for the background. If None then the mean surface brightness in the region outside "bkglim" is used. Relevant only if fit_bkg = True. Defaults to None.
    :type back: float
    :param samplefile: Name of ASCII file to output the final PyMC3 samples
    :type samplefile: str
    :param nrc: Number of core radii values to set up the multiscale model. Defaults to the number of data points / 4
    :type nrc: int
    :param nbetas: Number of beta values to set up the multiscale model (default = 6)
    :type nbetas: int
    :param min_beta: Minimum beta value (default = 0.6)
    :type min_beta: float
    :param nmore: Number of points to the define the fine grid onto which the mass model and the integration are performed, i.e. for one spectroscopic/SZ value, how many grid points will be defined. Defaults to 5.
    :type nmore: int
    :param tune: Number of NUTS tuning steps. Defaults to 500
    :type tune: int
    :param find_map: Specify whether a maximum likelihood fit will be performed first to initiate the sampler. Defaults to True
    :type find_map: bool
    """

    prof = Mhyd.sbprof
    sb = prof.profile
    esb = prof.eprof
    rad = prof.bins
    erad = prof.ebins
    counts = prof.counts
    area = prof.area
    exposure = prof.effexp
    bkgcounts = prof.bkgcounts

    # Define maximum radius for source deprojection, assuming we have only background for r>bkglim
    if bkglim is None:
        bkglim=np.max(rad+erad)
        Mhyd.bkglim = bkglim
        if back is None:
            back = sb[len(sb) - 1]
    else:
        Mhyd.bkglim = bkglim
        backreg = np.where(rad>bkglim)
        if back is None:
            back = np.mean(sb[backreg])

    # Set source region
    sourcereg = np.where(rad < bkglim)

    # Set vector with list of parameters
    pars = list_params(rad, sourcereg, nrc, nbetas, min_beta)

    npt = len(pars)

    if prof.psfmat is not None:
        psfmat = np.transpose(prof.psfmat)
    else:
        psfmat = np.eye(prof.nbin)

    # Compute linear combination kernel
    if fit_bkg:

        K = calc_linear_operator(rad, sourcereg, pars, area, exposure, psfmat) # transformation to counts

    else:

        Ksb = calc_sb_operator(rad, sourcereg, pars, withbkg=False)

        K = np.dot(prof.psfmat, Ksb)

    # Set up initial values
    if np.isnan(sb[0]) or sb[0] <= 0:
        testval = -10.
    else:
        testval = np.log(sb[0] / npt)
    if np.isnan(back) or back <= 0 or back is None:
        testbkg = -10.
    else:
        testbkg = np.log(back)

    z = Mhyd.redshift

    transf = 4. * (1. + z) ** 2 * (180. * 60.) ** 2 / np.pi / 1e-14 * Mhyd.nhc / cgsMpc * 1e3

    pardens = list_params_density(rad, sourcereg, Mhyd.amin2kpc, nrc, nbetas, min_beta)

    if fit_bkg:

        Kdens = calc_density_operator(rad, pardens, Mhyd.amin2kpc)

    else:

        Kdens = calc_density_operator(rad, pardens, Mhyd.amin2kpc, withbkg=False)

    # Define the fine grid onto which the mass model will be computed
    rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore=nmore)

    nptmore = len(rout_m)

    vx = MyDeprojVol(rin_m / Mhyd.amin2kpc, rout_m / Mhyd.amin2kpc)

    vol = vx.deproj_vol().T

    Mhyd.cf_prof = None

    try:
        nn = len(Mhyd.ccf)

    except TypeError:

        print('Single conversion factor provided, we will assume it is constant throughout the radial range')

        cf = Mhyd.ccf

    else:

        if len(Mhyd.ccf) != len(rad):

            print('The provided conversion factor has a different length as the input radial binning. Adopting the mean value.')

            cf = np.mean(Mhyd.ccf)

        else:

            print('Interpolating conversion factor profile onto the radial grid')

            cf = np.interp(rout_m, rad * Mhyd.amin2kpc, Mhyd.ccf)

            Mhyd.cf_prof = cf

    if Mhyd.spec_data is not None:

        if Mhyd.spec_data.psfmat is not None:

            mat1 = np.dot(Mhyd.spec_data.psfmat.T, sum_mat)

            proj_mat = np.dot(mat1, vol)

        else:

            proj_mat = np.dot(sum_mat, vol)

    if fit_bkg:

        Kdens_m = calc_density_operator(rout_m / Mhyd.amin2kpc, pardens, Mhyd.amin2kpc)

    else:

        Kdens_m = calc_density_operator(rout_m / Mhyd.amin2kpc, pardens, Mhyd.amin2kpc, withbkg=False)

    hydro_model = pm.Model()

    with hydro_model:
        # Priors for unknown model parameters
        coefs = pm.Normal('coefs', mu=testval, sigma=20, shape=npt)

        if fit_bkg:

            bkgd = pm.Normal('bkg', mu=testbkg, sigma=0.05, shape=1) # in case fit_bkg = False this is not fitted

            ctot = pm.math.concatenate((coefs, bkgd), axis=0)

            al = pm.math.exp(ctot)

            pred = pm.math.dot(K, al) + bkgcounts  # Predicted number of counts per annulus

        else:

            al = pm.math.exp(coefs)

            pred = pm.math.dot(K, al)


        # Model parameters
        allpmod = []

        for i in range(Forward.npar):

            name = Forward.parnames[i]

            if not Forward.fix[i]:

                lim = Forward.limits[i]

                if name == 'p0':

                    tpar = pm.TruncatedNormal(name, mu=np.log(Forward.start[i]), sigma=Forward.sd[i] / Forward.start[i],
                                                lower=np.log(lim[0]), upper=np.log(lim[1])) #log-normal prior on normalization

                    modpar = pm.math.exp(tpar)

                else:

                    modpar = pm.TruncatedNormal(name, mu=Forward.start[i], sigma=Forward.sd[i],
                                                lower=lim[0], upper=lim[1]) #Gaussian prior on other parameters
            else:

                dummy = pm.Normal('dummy'+name, mu=0., sigma=1.)

                dummy_param = 0 * dummy + Forward.start[i]

                modpar = pm.Deterministic(name, dummy_param)

            allpmod.append(modpar)

        pmod = pm.math.stack(allpmod, axis=0)

        dens_m = pm.math.sqrt(pm.math.dot(Kdens_m, al) / cf * transf)  # electron density in cm-3

        p3d = Forward.func_pm(rout_m, *pmod)

        # Density Likelihood
        if fit_bkg:

            count_obs = pm.Poisson('counts', mu=pred, observed=counts)  # counts likelihood

        else:

            sb_obs = pm.Normal('sb', mu=pred, observed=sb, sigma=esb)  # Sx likelihood

        # Temperature model and likelihood
        if Mhyd.spec_data is not None:

            # Model temperature
            t3d = p3d / dens_m

            # Mazzotta weights
            ei = dens_m ** 2 * t3d ** (-0.75)

            # Temperature projection
            flux = pm.math.dot(proj_mat, ei)

            tproj = pm.math.dot(proj_mat, t3d * ei) / flux

            T_obs = pm.Normal('kt', mu=tproj, observed=Mhyd.spec_data.temp_x, sigma=Mhyd.spec_data.errt_x)  # temperature likelihood

        # SZ pressure model and likelihood
        if Mhyd.sz_data is not None:
            pfit = p3d[index_sz]

            P_obs = pm.MvNormal('P', mu=pfit, observed=Mhyd.sz_data.pres_sz, cov=Mhyd.sz_data.covmat_sz)  # SZ pressure likelihood

    tinit = time.time()

    print('Running MCMC...')

    isjax = False

    try:
        import pymc.sampling.jax as pmjax

    except ImportError:
        print('JAX not found, using default sampler')

    else:
        isjax = True
        import pymc.sampling.jax as pmjax

    with hydro_model:

        if find_map:

            start = pm.find_MAP()

            if not isjax:

                trace = pm.sample(nmcmc, init='ADVI', initvals=start, tune=tune, return_inferencedata=True, target_accept=0.9)

            else:

                trace = pmjax.sample_numpyro_nuts(nmcmc, init='ADVI', initvals=start, tune=tune, return_inferencedata=True,
                                  target_accept=0.9)

        else:

            if not isjax:

                trace = pm.sample(nmcmc, tune=tune, init='ADVI',  return_inferencedata=True, target_accept=0.9)

            else:

                trace = pmjax.sample_numpyro_nuts(nmcmc, tune=tune, return_inferencedata=True, target_accept=0.9)

    print('Done.')

    tend = time.time()

    print(' Total computing time is: ', (tend - tinit) / 60., ' minutes')

    Mhyd.trace = trace

    # Get chains and save them to file
    chain_coefs = np.array(trace.posterior['coefs'])

    sc_coefs = chain_coefs.shape

    sampc = chain_coefs.reshape(sc_coefs[0] * sc_coefs[1], sc_coefs[2])

    if fit_bkg:

        sampb = np.array(trace.posterior['bkg']).flatten()

        samples = np.append(sampc, sampb, axis=1)

    else:
        samples = sampc

    Mhyd.samples = samples

    if samplefile is not None:
        np.savetxt(samplefile, samples)
        np.savetxt(samplefile + '.par', np.array([pars.shape[0] / nbetas, nbetas, min_beta, nmcmc]), header='pymc3')

    # Compute output deconvolved brightness profile

    if fit_bkg:
        Ksb = calc_sb_operator(rad, sourcereg, pars)

        allsb = np.dot(Ksb, np.exp(samples.T))

        bfit = np.median(np.exp(samples[:, npt]))

        Mhyd.bkg = bfit

        allsb_conv = np.dot(prof.psfmat, allsb[:, :npt])

    else:
        Ksb = calc_sb_operator(rad, sourcereg, pars, withbkg=False)

        allsb = np.dot(Ksb, np.exp(samples.T))

        allsb_conv = np.dot(K, np.exp(samples.T))

    pmc = np.median(allsb, axis=1)
    pmcl = np.percentile(allsb, 50. - 68.3 / 2., axis=1)
    pmch = np.percentile(allsb, 50. + 68.3 / 2., axis=1)
    Mhyd.sb_dec = pmc
    Mhyd.sb_dec_lo = pmcl
    Mhyd.sb_dec_hi = pmch

    pmc = np.median(allsb_conv, axis=1)
    pmcl = np.percentile(allsb_conv, 50. - 68.3 / 2., axis=1)
    pmch = np.percentile(allsb_conv, 50. + 68.3 / 2., axis=1)
    Mhyd.sb = pmc
    Mhyd.sb_lo = pmcl
    Mhyd.sb_hi = pmch

    Mhyd.nrc = nrc
    Mhyd.nbetas = nbetas
    Mhyd.min_beta = min_beta
    Mhyd.nmore = nmore
    Mhyd.pardens = pardens
    Mhyd.fit_bkg = fit_bkg

    alldens = np.sqrt(np.dot(Kdens, np.exp(samples.T)) * transf)
    pmc = np.median(alldens, axis=1) / np.sqrt(Mhyd.ccf)
    pmcl = np.percentile(alldens, 50. - 68.3 / 2., axis=1) / np.sqrt(Mhyd.ccf)
    pmch = np.percentile(alldens, 50. + 68.3 / 2., axis=1) / np.sqrt(Mhyd.ccf)
    Mhyd.dens = pmc
    Mhyd.dens_lo = pmcl
    Mhyd.dens_hi = pmch

    samppar = np.empty((len(samples), Forward.npar))
    for i in range(Forward.npar):

        name = Forward.parnames[i]

        if name == 'p0':

            samppar[:, i] = np.exp(np.array(trace.posterior[name]).flatten())

        else:

            samppar[:, i] = np.array(trace.posterior[name]).flatten()

    Mhyd.samppar = samppar

    Mhyd.K = K
    Mhyd.Kdens = Kdens
    Mhyd.Ksb = Ksb
    Mhyd.transf = transf
    Mhyd.Kdens_m = Kdens_m

    if Mhyd.spec_data is not None:
        kt_mod = kt_forw_from_samples(Mhyd, Forward, nmore=nmore)
        Mhyd.ktmod = kt_mod['TSPEC']
        Mhyd.ktmod_lo = kt_mod['TSPEC_LO']
        Mhyd.ktmod_hi = kt_mod['TSPEC_HI']
        Mhyd.kt3d = kt_mod['T3D']
        Mhyd.kt3d_lo = kt_mod['T3D_LO']
        Mhyd.kt3d_hi = kt_mod['T3D_HI']

    if Mhyd.sz_data is not None:
        pmed, plo, phi = P_forw_from_samples(Mhyd, Forward, nmore=nmore)
        Mhyd.pmod = pmed
        Mhyd.pmod_lo = plo
        Mhyd.pmod_hi = phi
