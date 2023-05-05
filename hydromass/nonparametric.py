import numpy as np
from .deproject import *
from .plots import rads_more, get_coolfunc, estimate_P0, plt
from scipy.interpolate import interp1d
import pymc as pm

# Function to compute linear operator transforming norms of GP model into radial profile

def calc_gp_operator(npt, rads, rin, rout, bin_fact=1.0, smin=None, smax=None):
    '''
    Set up a linear operator transforming the model coefficients into the temperature profile as a sum of Gaussian functions,

    .. math::

        T(r) = \\sum_{i=1}^P N_i G_i(r)

    with :math:`N_i` the model coefficients and :math:`G_i(r)` the defined Gaussian functions outputted by this task,

    .. math::

        G_i(r) = \\frac{1}{\\sqrt{2\\pi}\\sigma_i}\\exp \\left( \\frac{(r - \\mu_{i})^2}{2\\sigma_i^2} \\right)

    :param npt: Number of Gaussians to decompose the model into
    :type npt: int
    :param rads: Profile radii at which the model will be evaluated
    :type rads: numpy.ndarray
    :param rin: Inner boundaries of the profile radii
    :type rin: numpy.ndarray
    :param rout: Outer boundaries of the profile radii
    :type rout: numpy.ndarray
    :param bin_fact: Binning factor for the definition of the Gaussian standard deviations, i.e. the standard deviations of the Gaussians will be set to bin_fact * (rout - rin). The larger the value of bin_fact the stronger the smoothing, but the less accurate and flexible the model. Defaults to 1.
    :type bin_fact: float
    :param smin: Minimum value of the Gaussian standard deviation. If None, the width of the bins will be used (see bin_fact). If the value is set, the smoothing scales are set as logarithmically spaced between smin and smax. Defaults to None.
    :type smin: float
    :param smax: Maximum value of the Gaussian standard deviation. If None, the width of the bins will be used (see bin_fact). If the value is set, the smoothing scales are set as logarithmically spaced between smin and smax. Defaults to None.
    :type smax: float
    :return:
        - rg: Linear operator (2D array)
        - rgaus: Mean radii of the Gaussians (1D array)
        - sig: Standard deviations of the Gaussians (1D array)
    '''
    # Set up the Gaussian Process model
    rmin = (rin[0] + rout[0]) / 2.

    rmax = np.max(rout)

    width = rout - rin

    # Gaussians logarithmically spaced between min and max radius
    rgaus = np.logspace(np.log10(rmin), np.log10(rmax), npt)

    if smin is None or smax is None:

        # Sigma logarithmically spaced between min and max bin size
        sig = np.logspace(np.log10(bin_fact * rmin), np.log10(np.max(bin_fact * width)),npt)

    else:

        # Sigma logarithmically spaced between min and max bin size
        sig = np.logspace(np.log10(smin), np.log10(smax), npt)

    nrads = len(rads)  # rads may or may not be equal to bins

    # Extend into 2D and compute values of Gaussians at each point
    sigext = np.tile(sig, nrads).reshape(nrads, npt)

    rgext = np.tile(rgaus, nrads).reshape(nrads, npt)

    radsext = np.repeat(rads, npt).reshape(nrads, npt)

    rg = 1. / (np.sqrt(2. * np.pi) * sigext) * np.exp(-(radsext - rgext) ** 2 / 2. / sigext ** 2)

    return rg , rgaus, sig

def calc_gp_operator_lognormal(npt, rads, rin, rout, bin_fact=1.0, smin=None, smax=None):
    '''
    Same as :func:`hydromass.nonparametric.calc_gp_operator` with log-normal functions instead of Gaussians

    .. math::

        G_i(r) = \\frac{1}{\\sqrt{2\\pi}\\sigma_i}\\exp \\left( \\frac{(\\ln(r) - \\ln(\\mu_{i}))^2}{2\\sigma_i^2} \\right)

    :param npt: Number of log-normal functions to decompose the model into
    :type npt: int
    :param rads: Profile radii at which the model will be evaluated
    :type rads: numpy.ndarray
    :param rin: Inner boundaries of the profile radii
    :type rin: numpy.ndarray
    :param rout: Outer boundaries of the profile radii
    :type rout: numpy.ndarray
    :param bin_fact: Binning factor for the definition of the log-normal standard deviations, i.e. the standard deviations of the Gaussians will be set to bin_fact * (rout - rin). The larger the value of bin_fact the stronger the smoothing, but the less accurate and flexible the model. Defaults to 1.
    :type bin_fact: float
    :param smin: Minimum value of the log-normal standard deviation. If None, the width of the bins will be used (see bin_fact). If the value is set, the smoothing scales are set as logarithmically spaced between smin and smax. Defaults to None.
    :type smin: float
    :param smax: Maximum value of the log-normal standard deviation. If None, the width of the bins will be used (see bin_fact). If the value is set, the smoothing scales are set as logarithmically spaced between smin and smax. Defaults to None.
    :type smax: float
    :return:
        - rg: Linear operator (2D array)
        - rgaus: Mean radii of the log-normals (1D array)
        - sig: Standard deviations of the log-normals (1D array)
    '''

    # Set up the Gaussian Process model
    rmin = (rin[0] + rout[0]) / 2.

    rmax = np.max(rout)

    width = rout - rin

    # Gaussians logarithmically spaced between min and max radius
    rgaus = np.linspace(np.log(rmin/2.), np.log(rmax*2.), npt)
    #rgaus = np.linspace(np.log(rmin), np.log(rmax), npt)
    #outside = np.where(np.logical_or(rgaus<np.log(rmin), rgaus>np.log(rmax)))

    if smin is None or smax is None:

        # Sigma logarithmically spaced between min and max bin size
        sig = np.exp(np.linspace(np.log(bin_fact * rmin), np.log(np.max(bin_fact * width)),npt)) / np.exp(rgaus)
        #sig = np.ones(npt) * bin_fact * np.mean(width) / np.mean(rin)

    else:
        # Sigma logarithmically spaced between min and max bin size
        sig = np.ones(npt) * smin / np.mean(rin)

    #sig[outside] = 2.0

    nrads = len(rads)  # rads may or may not be equal to bins

    # Extend into 2D and compute values of Gaussians at each point
    sigext = np.tile(sig, nrads).reshape(nrads, npt)

    rgext = np.tile(rgaus, nrads).reshape(nrads, npt)

    radsext = np.log(np.repeat(rads, npt).reshape(nrads, npt))

    rg = 1. / (np.sqrt(2. * np.pi) * sigext) * np.exp(-(radsext - rgext) ** 2 / 2. / sigext ** 2)

    return rg , rgaus, sig


# Analytical gradient of the Gaussian process model

def calc_gp_grad_operator(npt, rads, rin, rout, bin_fact=1.0, smin=None, smax=None):
    '''
    Compute a linear operator transforming a parameter vector into a temperature gradient profile for the Gaussian mixture model

    :param npt: Number of Gaussians to decompose the model into
    :type npt: int
    :param rads: Profile radii at which the model will be evaluated
    :type rads: numpy.ndarray
    :param rin: Inner boundaries of the profile radii
    :type rin: numpy.ndarray
    :param rout: Outer boundaries of the profile radii
    :type rout: numpy.ndarray
    :param bin_fact: Binning factor for the definition of the Gaussian standard deviations, i.e. the standard deviations of the Gaussians will be set to bin_fact * (rout - rin). The larger the value of bin_fact the stronger the smoothing, but the less accurate and flexible the model. Defaults to 1.
    :type bin_fact: float
    :param smin: Minimum value of the Gaussian standard deviation. If None, the width of the bins will be used (see bin_fact). If the value is set, the smoothing scales are set as logarithmically spaced between smin and smax. Defaults to None.
    :type smin: float
    :param smax: Maximum value of the Gaussian standard deviation. If None, the width of the bins will be used (see bin_fact). If the value is set, the smoothing scales are set as logarithmically spaced between smin and smax. Defaults to None.
    :type smax: float
    :return: Linear operator (2D array)
    :rtype: numpy.ndarray
    '''
    # Set up the Gaussian Process model
    rmin = (rin[0] + rout[0]) / 2.

    rmax = np.max(rout)

    width = rout - rin

    # Gaussians logarithmically spaced between min and max radius
    rgaus = np.logspace(np.log10(rmin), np.log10(rmax), npt)

    if smin is None or smax is None:

        # Sigma logarithmically spaced between min and max bin size
        sig = np.logspace(np.log10(bin_fact * rmin), np.log10(np.max(bin_fact * width)), npt)

    else:

        # Sigma logarithmically spaced between min and max bin size
        sig = np.logspace(np.log10(smin), np.log10(smax), npt)

    nrads = len(rads)  # rads may or may not be equal to bins

    # Extend into 2D and compute values of Gaussians at each point
    sigext = np.tile(sig, nrads).reshape(nrads, npt)

    rgext = np.tile(rgaus, nrads).reshape(nrads, npt)

    radsext = np.repeat(rads, npt).reshape(nrads, npt)

    rg = 1. / (np.sqrt(2. * np.pi) * sigext ** 3) * np.exp(- (radsext - rgext) ** 2 / 2. / sigext ** 2) * (- (radsext - rgext))

    return rg


# Analytical gradient of the Gaussian process model

def calc_gp_grad_operator_lognormal(npt, rads, rin, rout, bin_fact=1.0, smin=None, smax=None):
    '''
    Compute a linear operator transforming a parameter vector into a temperature gradient profile for the log-normal mixture model

    :param npt: Number of Gaussians to decompose the model into
    :type npt: int
    :param rads: Profile radii at which the model will be evaluated
    :type rads: numpy.ndarray
    :param rin: Inner boundaries of the profile radii
    :type rin: numpy.ndarray
    :param rout: Outer boundaries of the profile radii
    :type rout: numpy.ndarray
    :param bin_fact: Binning factor for the definition of the Gaussian standard deviations, i.e. the standard deviations of the Gaussians will be set to bin_fact * (rout - rin). The larger the value of bin_fact the stronger the smoothing, but the less accurate and flexible the model. Defaults to 1.
    :type bin_fact: float
    :param smin: Minimum value of the Gaussian standard deviation. If None, the width of the bins will be used (see bin_fact). If the value is set, the smoothing scales are set as logarithmically spaced between smin and smax. Defaults to None.
    :type smin: float
    :param smax: Maximum value of the Gaussian standard deviation. If None, the width of the bins will be used (see bin_fact). If the value is set, the smoothing scales are set as logarithmically spaced between smin and smax. Defaults to None.
    :type smax: float
    :return: Linear operator (2D array)
    :rtype: numpy.ndarray
    '''
    # Set up the Gaussian Process model
    rmin = (rin[0] + rout[0]) / 2.

    rmax = np.max(rout)

    width = rout - rin

    # Gaussians logarithmically spaced between min and max radius
    rgaus = np.linspace(np.log(rmin/2.), np.log(rmax*2.), npt)
    #rgaus = np.linspace(np.log(rmin), np.log(rmax), npt)
    #outside = np.where(np.logical_or(rgaus<np.log(rmin), rgaus>np.log(rmax)))

    if smin is None or smax is None:

        # Sigma logarithmically spaced between min and max bin size
        sig = np.exp(np.linspace(np.log(bin_fact * rmin), np.log(np.max(bin_fact * width)),npt)) / np.exp(rgaus)
        #sig = np.ones(npt) * bin_fact * np.mean(width) / np.mean(rin)

    else:
        # Sigma logarithmically spaced between min and max bin size
        sig = np.ones(npt) * smin / np.mean(rin)

    #sig[outside] = 2.0

    nrads = len(rads)  # rads may or may not be equal to bins

    # Extend into 2D and compute values of Gaussians at each point
    sigext = np.tile(sig, nrads).reshape(nrads, npt)

    rgext = np.tile(rgaus, nrads).reshape(nrads, npt)

    radsext = np.repeat(rads, npt).reshape(nrads, npt)

    grad_ana = 1. / np.sqrt(2. * np.pi) / (sigext ** 3) * np.exp(- (np.log(radsext) - rgext) ** 2 / 2. / sigext ** 2) * (
        - (np.log(radsext) - rgext)) / radsext

    return grad_ana


def kt_GP_from_samples(Mhyd, nmore=5):
    """
    Compute the model temperature profile from the output of a non-parametric reconstruction run, evaluated at reference X-ray spectral radii

    :param Mhyd: :class:`hydromass.mhyd.Mhyd` object including the result of the non-parametric reconstruction run
    :type Mhyd: :class:`hydromass.mhyd.Mhyd`
    :return: Median temperature, Lower 1-sigma percentile, Upper 1-sigma percentile
    :rtype: numpy.ndarray
    """

    if Mhyd.spec_data is None:
        print('No spectral data provided')

        return

    nsamp = len(Mhyd.samples)

    rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore=nmore)

    vx = MyDeprojVol(rin_m / Mhyd.amin2kpc, rout_m / Mhyd.amin2kpc)

    vol_x = vx.deproj_vol().T

    if Mhyd.spec_data.psfmat is not None:

        mat1 = np.dot(Mhyd.spec_data.psfmat.T, sum_mat)

        proj_mat = np.dot(mat1, vol_x)

    else:

        proj_mat = np.dot(sum_mat, vol_x)

    nvalm = len(rin_m)

    if Mhyd.cf_prof is not None:

        rref_m = (rin_m + rout_m) / 2.

        rad = Mhyd.sbprof.bins

        tcf = np.interp(rref_m, rad * Mhyd.amin2kpc, Mhyd.ccf)

        cf_prof = np.repeat(tcf, nsamp).reshape(nvalm, nsamp)

    else:

        cf_prof = Mhyd.ccf

    dens_m = np.sqrt(np.dot(Mhyd.Kdens_m, np.exp(Mhyd.samples.T)) / cf_prof * Mhyd.transf)

    t3d = np.dot(Mhyd.GPop, Mhyd.samppar.T)

    if Mhyd.extend:

        if np.max(rout_m) > rout_m[ntm - 1]:
            # Power law outside of the fitted range
            ne0 = dens_m[nvalm - 1, :]

            T0 = Mhyd.sampp0 / ne0

            Tspo = t3d[ntm - 1, :]

            rspo = rout_m[ntm - 1]

            r0 = rout_m[nvalm - 1]

            alpha = - np.log(Tspo / T0) / np.log(rspo / r0)

            nout = nvalm - ntm

            outspec = np.where(rout_m > rspo)

            Tspo_mul = np.tile(Tspo, nout).reshape(nout, nsamp)

            rout_mul = np.repeat(rout_m[outspec], nsamp).reshape(nout, nsamp)

            alpha_mul = np.tile(alpha, nout).reshape(nout, nsamp)

            t3d[outspec] = Tspo_mul * (rout_mul / rspo) ** (-alpha_mul)

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


def P_GP_from_samples(Mhyd, nmore=5):
    """
    Compute the model pressure profile from the output of a non-parametric reconstruction run, evaluated at the reference SZ radii

    :param Mhyd: :class:`hydromass.mhyd.Mhyd` object including the result of the non-parametric reconstruction run
    :type Mhyd: :class:`hydromass.mhyd.Mhyd`
    :return: Median pressure, Lower 1-sigma percentile, Upper 1-sigma percentile
    :rtype: numpy.ndarray
    """

    if Mhyd.sz_data is None:

        print('No SZ data provided')

        return

    rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore=nmore)

    nsamp = len(Mhyd.samples)

    nvalm = len(rin_m)

    if Mhyd.cf_prof is not None:

        rref_m = (rin_m + rout_m) / 2.

        rad = Mhyd.sbprof.bins

        tcf = np.interp(rref_m, rad * Mhyd.amin2kpc, Mhyd.ccf)

        cf_prof = np.repeat(tcf, nsamp).reshape(nvalm, nsamp)

    else:

        cf_prof = Mhyd.ccf

    dens_m = np.sqrt(np.dot(Mhyd.Kdens_m, np.exp(Mhyd.samples.T)) / cf_prof * Mhyd.transf)

    t3d = np.dot(Mhyd.GPop, Mhyd.samppar.T)

    if Mhyd.extend:

        if np.max(rout_m) > rout_m[ntm - 1]:
            # Power law outside of the fitted range
            ne0 = dens_m[nvalm - 1, :]

            T0 = Mhyd.sampp0 / ne0

            Tspo = t3d[ntm - 1, :]

            rspo = rout_m[ntm - 1]

            r0 = rout_m[nvalm - 1]

            alpha = - np.log(Tspo / T0) / np.log(rspo / r0)

            nout = nvalm - ntm

            outspec = np.where(rout_m > rspo)

            Tspo_mul = np.tile(Tspo, nout).reshape(nout, nsamp)

            rout_mul = np.repeat(rout_m[outspec], nsamp).reshape(nout, nsamp)

            alpha_mul = np.tile(alpha, nout).reshape(nout, nsamp)

            t3d[outspec] = Tspo_mul * (rout_mul / rspo) ** (-alpha_mul)

    p3d = t3d * dens_m

    pmt, plot, phit = np.percentile(p3d, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    pmed, plo, phi = pmt[index_sz], plot[index_sz], phit[index_sz]

    return pmed, plo, phi


def mass_GP_from_samples(Mhyd, rin=None, rout=None, npt=200, plot=False):
    '''
    Compute the hydrostatic mass profile from an existing non-parametric reconstruction run. The gradient of the basis functions for the temperature and the density are computed analytically and the best-fit coefficients are used to determine the posterior distributions of hydrostatic mass,

    .. math::

        M(<r) = -\\frac{r k T}{G \mu m_p}\\left[ \\frac{\\partial \\log T}{\\partial \\log r} + \\frac{\\partial \\log n_e}{\\partial \\log r} \\right]

    :param Mhyd: :class:`hydromass.mhyd.Mhyd` object including the result of the non-parametric reconstruction run
    :type Mhyd: :class:`hydromass.mhyd.Mhyd`
    :param rin: Minimum radius of the output profile. If None, the innermost data point is used. Defaults to None
    :type rin: float
    :param rout: Maximum radius of the output profile. If None, the outermost data point is used. Defaults to None
    :type rout: float
    :param npt: Number of points in output profile. Defaults to 200
    :type npt: int
    :param plot: If True, produce a plot of the mass profile and gas mass profile. Defaults to False
    :type plot: bool
    :return: Dictionary containing the profiles of hydrostatic mass, gas mass, and gas fraction
    :rtype: dict(11xnpt)
    '''

    nsamp = len(Mhyd.samples)

    rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore=Mhyd.nmore)

    if rin is None:
        rin = np.min((rin_m+rout_m)/2.)

        if rin == 0:
            rin = 1.

    if rout is None:
        rout = np.max(rout_m)

    bins = np.linspace(np.sqrt(rin), np.sqrt(rout), npt + 1)

    bins = bins ** 2

    rin_m = bins[:npt]

    rout_m = bins[1:]

    rref_m = (rin_m + rout_m) / 2.

    nvalm = len(rin_m)

    if Mhyd.cf_prof is not None:

        rref_m = (rin_m + rout_m) / 2.

        rad = Mhyd.sbprof.bins

        tcf = np.interp(rref_m, rad * Mhyd.amin2kpc, Mhyd.ccf)

        cf_prof = np.repeat(tcf, nsamp).reshape(nvalm, nsamp)

    else:

        cf_prof = Mhyd.ccf

    if Mhyd.fit_bkg:

        Kdens_m = calc_density_operator(rout_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc)

        Kdens_grad = calc_grad_operator(rout_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc)

    else:

        Kdens_m = calc_density_operator(rout_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc, withbkg=False)

        Kdens_grad = calc_grad_operator(rout_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc, withbkg=False)

    dens_m = np.sqrt(np.dot(Kdens_m, np.exp(Mhyd.samples.T)) / cf_prof * Mhyd.transf)

    grad_dens = np.dot(Kdens_grad, np.exp(Mhyd.samples.T)) / 2. / dens_m ** 2 / cf_prof * Mhyd.transf

    if Mhyd.spec_data is not None and Mhyd.sz_data is None:

        rout_joint = Mhyd.spec_data.rout_x

    elif Mhyd.spec_data is None and Mhyd.sz_data is not None:

        rout_joint = Mhyd.sz_data.rout_sz

    elif Mhyd.spec_data is not None and Mhyd.sz_data is not None:

        rout_joint = np.sort(np.append(Mhyd.spec_data.rout_x, Mhyd.sz_data.rout_sz))

    rin_joint = np.roll(rout_joint, 1)

    rin_joint[0] = 0.

    GPop, rgauss, sig = calc_gp_operator_lognormal(Mhyd.ngauss, rout_m, rin_joint, rout_joint, bin_fact=Mhyd.bin_fact, smin=Mhyd.smin, smax=Mhyd.smax)

    GPgrad = calc_gp_grad_operator_lognormal(Mhyd.ngauss, rout_m, rin_joint, rout_joint, bin_fact=Mhyd.bin_fact, smin=Mhyd.smin, smax=Mhyd.smax)

    t3d = np.dot(GPop, Mhyd.samppar.T)

    rout_mul = np.repeat(rout_m, nsamp).reshape(nvalm, nsamp) * cgskpc

    grad_t3d = rout_mul / cgskpc / t3d * np.dot(GPgrad, Mhyd.samppar.T)

    rspo = np.max(rout_joint)

    if Mhyd.extend:

        if rout > rspo:
            # Power law outside of the fitted range
            ne0 = dens_m[nvalm - 1, :]

            T0 = Mhyd.sampp0 / ne0

            finter = interp1d(rout_m, t3d, axis=0)

            Tspo = finter(rspo)

            r0 = rout_m[nvalm - 1]

            alpha = - np.log(Tspo / T0) / np.log(rspo / r0)

            outspec = np.where(rout_m > rspo)

            nout = len(outspec[0])

            Tspo_mul = np.tile(Tspo, nout).reshape(nout, nsamp)

            rout_mm = np.repeat(rout_m[outspec], nsamp).reshape(nout, nsamp)

            alpha_mul = np.tile(alpha, nout).reshape(nout, nsamp)

            t3d[outspec] = Tspo_mul * (rout_mm / rspo) ** (-alpha_mul)

            grad_t3d[outspec] = - alpha_mul

    mass = - rout_mul * t3d / (cgsG * cgsamu * Mhyd.mup) * (grad_t3d + grad_dens) * kev2erg / Msun

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


def prof_GP_hires(Mhyd, rin=None, npt=200, Z=0.3):
    """
    Compute the best-fitting thermodynamic profiles and error envelopes from a non-parametric reconstruction run. The output profiles include gas density, temperature, pressure, entropy, cooling function, and radiative cooling time.

    :param Mhyd: :class:`hydromass.mhyd.Mhyd` object including the result of the non-parametric reconstruction run
    :type Mhyd: :class:`hydromass.mhyd.Mhyd`
    :param rin: Minimum radius of the output profiles. If None, the innermost data point is used. Defaults to None
    :type rin: float
    :param npt: Number of points in the output profiles. Defaults to 200
    :type npt: int
    :param Z: Gas metallicity relative to Solar for the calculation of the cooling function. Defaults to 0.3
    :type Z: float
    :return: Dictionary including the thermodynamic profiles
    :rtype: dict(24xnpt)
    """

    rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore=Mhyd.nmore)

    if rin is None:
        rin = np.min(rin_m)

        if rin == 0:
            rin = 1.

    rout = np.max(rout_m)

    bins = np.linspace(np.sqrt(rin), np.sqrt(rout), npt + 1)

    bins = bins ** 2

    rin_m = bins[:npt]

    rout_m = bins[1:]

    rref_m = (rin_m + rout_m) / 2.

    vx = MyDeprojVol(rin_m / Mhyd.amin2kpc, rout_m / Mhyd.amin2kpc)

    vol_x = vx.deproj_vol().T

    nsamp = len(Mhyd.samples)

    nvalm = len(rin_m)

    if Mhyd.cf_prof is not None:

        rref_m = (rin_m + rout_m) / 2.

        rad = Mhyd.sbprof.bins

        tcf = np.interp(rref_m, rad * Mhyd.amin2kpc, Mhyd.ccf)

        cf_prof = np.repeat(tcf, nsamp).reshape(nvalm, nsamp)

    else:

        cf_prof = Mhyd.ccf

    if Mhyd.fit_bkg:

        Kdens_m = calc_density_operator(rout_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc)

    else:

        Kdens_m = calc_density_operator(rout_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc, withbkg=False)

    dens_m = np.sqrt(np.dot(Kdens_m, np.exp(Mhyd.samples.T)) / cf_prof * Mhyd.transf)

    if Mhyd.spec_data is not None and Mhyd.sz_data is None:

        rout_joint = Mhyd.spec_data.rout_x

    elif Mhyd.spec_data is None and Mhyd.sz_data is not None:

        rout_joint = Mhyd.sz_data.rout_sz

    elif Mhyd.spec_data is not None and Mhyd.sz_data is not None:

        rout_joint = np.sort(np.append(Mhyd.spec_data.rout_x, Mhyd.sz_data.rout_sz))

    rin_joint = np.roll(rout_joint, 1)

    rin_joint[0] = 0.

    GPop, rgauss, sig = calc_gp_operator_lognormal(Mhyd.ngauss, rout_m, rin_joint, rout_joint, bin_fact=Mhyd.bin_fact, smin=Mhyd.smin, smax=Mhyd.smax)

    t3d = np.dot(GPop, Mhyd.samppar.T)

    rspo = np.max(rout_joint)

    if Mhyd.extend:

        if rout > rspo:
            # Power law outside of the fitted range
            ne0 = dens_m[nvalm - 1, :]

            T0 = Mhyd.sampp0 / ne0

            finter = interp1d(rout_m, t3d, axis=0)

            Tspo = finter(rspo)

            r0 = rout_m[nvalm - 1]

            alpha = - np.log(Tspo / T0) / np.log(rspo / r0)

            outspec = np.where(rout_m > rspo)

            nout = len(outspec[0])

            Tspo_mul = np.tile(Tspo, nout).reshape(nout, nsamp)

            rout_mul = np.repeat(rout_m[outspec], nsamp).reshape(nout, nsamp)

            alpha_mul = np.tile(alpha, nout).reshape(nout, nsamp)

            t3d[outspec] = Tspo_mul * (rout_mul / rspo) ** (-alpha_mul)

    p3d = t3d * dens_m

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
        "R_REF": rref_m,
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



def Run_NonParametric_PyMC3(Mhyd, bkglim=None, nmcmc=1000, fit_bkg=False, back=None,
                   samplefile=None, nrc=None, nbetas=6, min_beta=0.6, nmore=5,
                   tune=500, bin_fact=1.0, smin=None, smax=None, ngauss=100, find_map=True,
                   extend=False):
    """
    Run non-parametric log-normal mixture reconstruction. Following Eckert et al. (2022), the temperature profile is described as a linear combination of a large number of log-normal functions, whereas the gas density profile is decomposed on a basis of King functions. The number of log-normal functions as well as the smoothing scales can be adjusted by the user.

    :param Mhyd: An input :class:`hydromass.mhyd.Mhyd` object including the data and the source definition
    :type Mhyd: :class:`hydromass.mhyd.Mhyd`
    :param bkglim: Radius (in arcmin) beyond which it is assumed that the background fully dominates the profile. If None, the entire radial range is fitted as source + background. Defaults to None.
    :type bkglim: float
    :param nmcmc: Number of NUTS samples. Defaults to 1000
    :type nmcmc: int
    :param fit_bkg: Define whether we will fit the counts as source + background (True) or the background subtracted surface brightness as source only (False). Defaults to False.
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
    :param bin_fact: bin_fact: Binning factor for the definition of the log-normal standard deviations, i.e. the standard deviations of the log-normals will be set to bin_fact * (rout - rin). The larger the value of bin_fact the stronger the smoothing, but the less accurate and flexible the model. Defaults to 1.
    :type bin_fact: float
    :param smin: Minimum value of the log-normal standard deviation. If None, the width of the bins will be used (see bin_fact). If the value is set, the smoothing scales are set as logarithmically spaced between smin and smax. Defaults to None.
    :type smin: float
    :param smax: Maximum value of the log-normal standard deviation. If None, the width of the bins will be used (see bin_fact). If the value is set, the smoothing scales are set as logarithmically spaced between smin and smax. Defaults to None.
    :type smax: float
    :param ngauss: Number of log-normal functions. Defaults to 100
    :type ngauss: int
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
        psfmat = prof.psfmat
    else:
        psfmat = np.eye(prof.nbin)

    # Compute linear combination kernel
    if fit_bkg:

        K = calc_linear_operator(rad, sourcereg, pars, area, exposure, np.transpose(psfmat)) # transformation to counts

    else:

        Ksb = calc_sb_operator(rad, sourcereg, pars, withbkg=False)

        K = np.dot(psfmat, Ksb)

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

    transf = Mhyd.transf

    pardens = list_params_density(rad, sourcereg, Mhyd.amin2kpc, nrc, nbetas, min_beta)

    if fit_bkg:

        Kdens = calc_density_operator(rad, pardens, Mhyd.amin2kpc)

    else:

        Kdens = calc_density_operator(rad, pardens, Mhyd.amin2kpc, withbkg=False)

    # Define the fine grid onto which the mass model will be computed
    rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore=nmore)

    rref_m = (rin_m + rout_m) / 2.

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

            cf = np.interp(rref_m, rad * Mhyd.amin2kpc, Mhyd.ccf)

            Mhyd.cf_prof = cf

    if Mhyd.spec_data is not None:

        if Mhyd.spec_data.psfmat is not None:

            mat1 = np.dot(Mhyd.spec_data.psfmat.T, sum_mat)

            proj_mat = np.dot(mat1, vol)

        else:

            proj_mat = np.dot(sum_mat, vol)

    if fit_bkg:

        Kdens_m = calc_density_operator(rout_m / Mhyd.amin2kpc, pardens, Mhyd.amin2kpc)

        Kdens_grad = calc_grad_operator(rout_m / Mhyd.amin2kpc, pardens, Mhyd.amin2kpc)

    else:

        Kdens_m = calc_density_operator(rout_m / Mhyd.amin2kpc, pardens, Mhyd.amin2kpc, withbkg=False)

        Kdens_grad = calc_grad_operator(rout_m / Mhyd.amin2kpc, pardens, Mhyd.amin2kpc, withbkg=False)

    if Mhyd.spec_data is not None and Mhyd.sz_data is None:

        rout_joint = Mhyd.spec_data.rout_x

    elif Mhyd.spec_data is None and Mhyd.sz_data is not None:

        rout_joint = Mhyd.sz_data.rout_sz

    elif Mhyd.spec_data is not None and Mhyd.sz_data is not None:

        rout_joint = np.sort(np.append(Mhyd.spec_data.rout_x, Mhyd.sz_data.rout_sz))

    rin_joint = np.roll(rout_joint, 1)

    rin_joint[0] = 0.

    GPop, rgauss, sig = calc_gp_operator_lognormal(ngauss, rout_m, rin_joint, rout_joint, bin_fact=bin_fact, smin=smin, smax=smax)

    GPgrad = calc_gp_grad_operator_lognormal(ngauss, rout_m, rin_joint, rout_joint, bin_fact=bin_fact, smin=smin, smax=smax)

    hydro_model = pm.Model()

    P0_est, err_P0_est = None, None

    if extend:

        P0_est = estimate_P0(Mhyd)

        err_P0_est = P0_est  # 1-dex

        Mhyd.extend = True

    else:
        Mhyd.extend = False

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

        # GP parameters
        #coefs_GP = pm.Normal('GP', mu=np.log(30./ngauss), sigma=20, shape=ngauss)
        coefs_GP = pm.Normal('GP', mu=1./np.sqrt(np.arange(1,ngauss+1)), sigma=20, shape=ngauss)

        # Expected value of outcome
        gpp = pm.math.exp(coefs_GP)

        t3d = pm.math.dot(GPop, gpp)

        dens_m = pm.math.sqrt(pm.math.dot(Kdens_m, al) / cf * transf)  # electron density in cm-3

        if extend:
            logp0 = pm.TruncatedNormal('logp0', mu=np.log(P0_est), sigma=err_P0_est / P0_est,
                                       lower=np.log(P0_est) - err_P0_est / P0_est,
                                       upper=np.log(P0_est) + err_P0_est / P0_est)

            if np.max(rout_m) > rout_m[ntm - 1]:
                # Power law outside of the fitted range
                ne0 = dens_m[nptmore - 1]

                T0 = np.exp(logp0) / ne0

                Tspo = t3d[ntm - 1]

                rspo = rout_m[ntm - 1]

                r0 = rout_m[nptmore - 1]

                alpha = - pm.math.log(Tspo/T0) / np.log(rspo/r0)

                outspec = np.where(rout_m > rspo)

                inspec = np.where(rout_m <= rspo)

                t3d_in = t3d[inspec]

                t3d_out = Tspo * (rout_m[outspec] / rspo) ** (-alpha)

                t3d = pm.math.concatenate([t3d_in, t3d_out])


        # Density Likelihood
        if fit_bkg:

            count_obs = pm.Poisson('counts', mu=pred, observed=counts)  # counts likelihood

        else:

            sb_obs = pm.Normal('sb', mu=pred, observed=sb, sigma=esb)  # Sx likelihood

        # Temperature model and likelihood
        if Mhyd.spec_data is not None:

            # Mazzotta weights
            ei = dens_m ** 2 * t3d ** (-0.75)

            # Temperature projection
            flux = pm.math.dot(proj_mat, ei)

            tproj = pm.math.dot(proj_mat, t3d * ei) / flux

            T_obs = pm.Normal('kt', mu=tproj, observed=Mhyd.spec_data.temp_x, sigma=Mhyd.spec_data.errt_x)  # temperature likelihood

        # SZ pressure model and likelihood
        if Mhyd.sz_data is not None:

            p3d = t3d * dens_m

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

                trace = pm.sample(nmcmc, initvals=start, tune=tune, return_inferencedata=True)

            else:

                trace = pmjax.sample_numpyro_nuts(nmcmc, initvals=start, tune=tune, return_inferencedata=True)

        else:

            if not isjax:

                trace = pm.sample(nmcmc, tune=tune)

            else:

                trace = pmjax.sample_numpyro_nuts(nmcmc, tune=tune)

    print('Done.')

    tend = time.time()

    print(' Total computing time is: ', (tend - tinit) / 60., ' minutes')

    Mhyd.trace = trace

    # Get chains and save them to file
    chain_coefs = np.array(trace.posterior['coefs'])

    sc_coefs = chain_coefs.shape

    sampc = chain_coefs.reshape(sc_coefs[0]*sc_coefs[1], sc_coefs[2])

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

    # Get chains and save them to file
    chain_gp = np.exp(np.array(trace.posterior['GP']))

    sc_gp = chain_gp.shape

    samppar = chain_gp.reshape(sc_gp[0]*sc_gp[1], sc_gp[2])

    Mhyd.samppar = samppar

    Mhyd.GPop = GPop
    Mhyd.GPgrad = GPgrad
    Mhyd.smin = smin
    Mhyd.smax = smax
    Mhyd.bin_fact = bin_fact
    Mhyd.ngauss = ngauss

    Mhyd.K = K
    Mhyd.Kdens = Kdens
    Mhyd.Ksb = Ksb
    Mhyd.transf = transf
    Mhyd.Kdens_m = Kdens_m
    Mhyd.Kdens_grad = Kdens_grad

    if extend:
        sampp0 = np.exp(trace.posterior['logp0'].to_numpy().flatten())
        Mhyd.sampp0 = sampp0

    if Mhyd.spec_data is not None:
        kt_mod = kt_GP_from_samples(Mhyd, nmore=nmore)
        Mhyd.ktmod = kt_mod['TSPEC']
        Mhyd.ktmod_lo = kt_mod['TSPEC_LO']
        Mhyd.ktmod_hi = kt_mod['TSPEC_HI']
        Mhyd.kt3d = kt_mod['T3D']
        Mhyd.kt3d_lo = kt_mod['T3D_LO']
        Mhyd.kt3d_hi = kt_mod['T3D_HI']

    if Mhyd.sz_data is not None:
        pmed, plo, phi = P_GP_from_samples(Mhyd, nmore=nmore)
        Mhyd.pmod = pmed
        Mhyd.pmod_lo = plo
        Mhyd.pmod_hi = phi
