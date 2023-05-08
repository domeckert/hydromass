import numpy as np
from .deproject import *
from .plots import rads_more, get_coolfunc, plt
from .functions import ArcTan
import pymc as pm

tt_arctan = ArcTan()

# Gamma(R) function
def func_poly_rad_pm(x, pars, dens, grad_dens):
    '''
    Theano model of the effective polytropic index as a function of radius following Ghirardini et al. (2019), modified to avoid the divergence in the core.

    .. math::

        \\Gamma(r) = 1 + \\Gamma_0 + \\Gamma_R \\frac{\\partial \\log R}{\\partial \\log n_e} \\frac{1}{1 + r/Rs}

    The function returns the model pressure as

    .. math::

        P_{3D}(r) =  P_0 n_e(r) ^ {\\Gamma(r)}

    :param x: Scaled radius
    :param pars: Parameters of the radial polytropic index model
    :param dens: Density profile
    :param grad_dens: Logarithmic density gradient
    :return: Pressure
    '''

    p0 = pars[0]

    Gamma0 = pars[1]

    GammaR = pars[2]

    Rs = pars[3]

    poly_index = 1 + Gamma0 + GammaR / grad_dens / (1. + x / Rs)

    p3d = p0 * dens ** poly_index

    return p3d

def func_poly_rad_np(xout, pars, dens, grad_dens):
    '''
    Numpy function for the radial polytropic index model (see :func:`hydromass.polytropic.func_poly_rad_pm`)

    :param xout: Scaled radius
    :type xout: numpy.ndarray
    :param pars: List of parameters of the radial polytropic index model
    :type pars: numpy.ndarray
    :param dens: Density profile realizations
    :type dens: numpy.ndarray
    :param grad_dens: Logarithmic density gradient realizations
    :type grad_dens: numpy.ndarray
    :return: Pressure profile samples
    :rtype: numpy.ndarray
    '''

    p0 = pars[:, 0]

    Gamma0 = pars[:, 1]

    GammaR = pars[:, 2]

    Rs =pars[:, 3]

    npt = len(xout)

    npars = len(p0)

    p0mul = np.repeat(p0, npt).reshape(npars, npt)

    Gamma0mul = np.repeat(Gamma0, npt).reshape(npars, npt)

    GammaRmul = np.repeat(GammaR, npt).reshape(npars, npt)

    Rsmul = np.repeat(Rs, npt).reshape(npars, npt)

    xoutmul = np.tile(xout, npars).reshape(npars, npt)

    poly_index = 1. + Gamma0mul + GammaRmul / grad_dens.T / (1. + xoutmul / Rsmul)

    p3d = p0mul * dens.T ** poly_index

    return p3d.T

# Gamma(n) function
def func_poly_dens_pm(x, pars, dens, grad_dens):
    '''
    Theano model of the effective polytropic index as a function of gas density

    .. math::

        \\Gamma(n_e) = \\Gamma_0 \\left[ 1 + \\Gamma_S \\arctan\\left( \\frac{\\log(n_e E(z)^{-2}/n_0)}{\\Gamma_T} \\right) \\right]

    The function returns the model pressure as

    .. math::

        P_{3D} = P_0 \\left( \\frac{n_e E(z)^{-2}}{n_0} \\right)^{\\Gamma(n_e)}

    :param x: Radius
    :param pars: Parameter vector of the polytropic model (:math:`P_0`, :math:`n_0`, :math:`\\Gamma_T`, :math:`\\Gamma_0`, and :math:`\\Gamma_S`)
    :param dens: Density profile
    :param grad_dens: Logarithmic density gradient
    :return: Pressure profile
    '''

    p0 = pars[0]

    logn0 = pars[1]

    trans = pars[2]

    Gamma0 = pars[3]

    scale = pars[4]

    logne = pm.math.log(dens)

    poly_index = Gamma0 * (1. + scale * tt_arctan((logne - logn0) / trans))

    p3d = p0 * pm.math.exp(poly_index * (logne - logn0))

    return  p3d


def func_poly_dens_np(xout, pars, dens, grad_dens):
    '''
    Numpy function for the density dependent polytropic index model (see :func:`hydromass.polytropic.func_poly_dens_pm`)

    :param xout: Scaled radius
    :type xout: numpy.ndarray
    :param pars: List of parameters of the density dependent polytropic index model
    :type pars: numpy.ndarray
    :param dens: Density profile realizations
    :type dens: numpy.ndarray
    :param grad_dens: Logarithmic density gradient realizations
    :type grad_dens: numpy.ndarray
    :return: Pressure profile samples
    :rtype: numpy.ndarray
    '''

    p0 = pars[:, 0]

    logn0 = pars[:, 1]

    trans = pars[:, 2]

    Gamma0 = pars[:, 3]

    scale = pars[:, 4]

    npt = len(xout)

    npars = len(p0)

    p0mul = np.repeat(p0, npt).reshape(npars, npt)

    logn0mul = np.repeat(logn0, npt).reshape(npars, npt)

    transmul = np.repeat(trans, npt).reshape(npars, npt)

    Gamma0mul = np.repeat(Gamma0, npt).reshape(npars, npt)

    scalemul = np.repeat(scale, npt).reshape(npars, npt)

    poly_index = Gamma0mul * (1. + scalemul * np.arctan((np.log(dens.T) - logn0mul) / transmul))

    p3d = p0mul * np.exp(poly_index * (np.log(dens.T) - logn0mul))

    return  p3d.T


# Pressure gradient from Gamma(n) function
def gradP_gamman(xout, pars, dens, grad_dens):
    '''
    Numpy function for the pressure gradient in the density dependent polytropic index model

    .. math::

        \\frac{d\\log P}{d\\log r} = \\left[ \\Gamma + \\log(n)\\frac{d\\Gamma}{d\\log n} \\right]\\frac{\\partial \\log n}{\\partial \\log r}

    with

    .. math::

        \\frac{d\\Gamma}{d\\log n} = \\frac{\\Gamma_0 \\Gamma_S}{\\Gamma_T}\\left[ 1 + \\left(\\frac{\\log(n/n_0)}{\\Gamma_T}\\right)^2 \\right]^{-1}

    :param xout: Scaled radius
    :type xout: numpy.ndarray
    :param pars: List of parameters of the density dependent polytropic index model
    :type pars: numpy.ndarray
    :param dens: Density profile realizations
    :type dens: numpy.ndarray
    :param grad_dens: Logarithmic density gradient realizations
    :type grad_dens: numpy.ndarray
    :return: Pressure gradient samples
    :rtype: numpy.ndarray
    '''

    logn0 = pars[:, 1]

    trans = pars[:, 2]

    Gamma0 = pars[:, 3]

    scale = pars[:, 4]

    npt = len(xout)

    npars = len(logn0)

    logne = np.log(dens.T)

    logn0mul = np.repeat(logn0, npt).reshape(npars, npt)

    transmul = np.repeat(trans, npt).reshape(npars, npt)

    Gamma0mul = np.repeat(Gamma0, npt).reshape(npars, npt)

    scalemul = np.repeat(scale, npt).reshape(npars, npt)

    poly_index = Gamma0mul * (1. + scalemul * np.arctan((logne - logn0mul) / transmul))

    gamma_grad = Gamma0mul * scalemul / transmul * (1. + ((logne -logn0mul) / transmul) ** 2) ** (-1)

    dpdn = (logne - logn0mul) * gamma_grad + poly_index

    return dpdn.T * grad_dens


def kt_poly_from_samples(Mhyd, Polytropic, nmore=5):
    """
    Compute the model temperature profile from an existing polytropic reconstruction evaluated at reference X-ray temperature radii

    :param Mhyd: :class:`hydromass.mhyd.Mhyd` object including the polytropic reconstruction results
    :type Mhyd: :class:`hydromass.mhyd.Mhyd`
    :param Polytropic: Polytropic model defined using the :class:`hydromass.polytropic.Polytropic` class
    :type Polytropic: :class:`hydromass.polytropic.Polytropic`
    :param nmore: Number of subgrid values to compute the fine binning. Each input bin will be split into nmore values. Defaults to 5.
    :type nmore: int
    :return: Dictionary including the 3D and spectroscopic-like model temperatures and their uncertainties
    :rtype: dict(9xnpt)
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

    npx = len(Mhyd.spec_data.rref_x)

    dens_m = np.sqrt(np.dot(Mhyd.Kdens_m, np.exp(Mhyd.samples.T)) / Mhyd.ccf * Mhyd.transf)

    grad_dens = np.dot(Mhyd.Kdens_grad, np.exp(Mhyd.samples.T)) / 2. / dens_m ** 2 / Mhyd.ccf * Mhyd.transf

    p3d = Polytropic.func_np(rout_m, Mhyd.samppar, dens_m, grad_dens)

    t3d = p3d / dens_m

    # Mazzotta weights
    ei = dens_m ** 2 * t3d ** (-0.75)

    # Temperature projection
    flux = np.dot(proj_mat, ei)

    tproj = np.dot(proj_mat, t3d * ei) / flux

    t3d_or = np.empty((npx, nsamp))

    for i in range(nsamp):

        t3d_or[:, i] = t3d[:, i][index_x]

    tmed, tlo, thi = np.percentile(tproj, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    t3do, t3dl, t3dh = np.percentile(t3d_or,[50., 50. - 68.3 / 2. , 50. + 68.3 / 2.] , axis=1)

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

def P_poly_from_samples(Mhyd, Polytropic, nmore=5):
    """
    Compute the model pressure profile from an existing polytropic reconstruction evaluated at the reference SZ radii

    :param Mhyd: :class:`hydromass.mhyd.Mhyd` object including the polytropic reconstruction results
    :type Mhyd: :class:`hydromass.mhyd.Mhyd`
    :param Polytropic: Polytropic model defined using the :class:`hydromass.polytropic.Polytropic` class
    :type Polytropic: :class:`hydromass.polytropic.Polytropic`
    :param nmore: Number of subgrid values to compute the fine binning. Each input bin will be split into nmore values. Defaults to 5.
    :type nmore: int
    :return: Median pressure, Lower 1-sigma percentile, Upper 1-sigma percentile
    :rtype: numpy.ndarray
    """

    if Mhyd.sz_data is None:

        print('No SZ data provided')

        return

    nsamp = len(Mhyd.samples)

    rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore=nmore)

    npx = len(Mhyd.sz_data.rref_sz)

    dens_m = np.sqrt(np.dot(Mhyd.Kdens_m, np.exp(Mhyd.samples.T)) / Mhyd.ccf * Mhyd.transf)

    grad_dens = np.dot(Mhyd.Kdens_grad, np.exp(Mhyd.samples.T)) / 2. / dens_m ** 2 / Mhyd.ccf * Mhyd.transf

    p3d = Polytropic.func_np(rout_m, Mhyd.samppar, dens_m, grad_dens)

    pfit = np.empty((npx, nsamp))

    for i in range(nsamp):

        pfit[:, i] = p3d[:, i][index_sz]

    pmed, plo, phi = np.percentile(pfit, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    return pmed, plo, phi


def mass_poly_from_samples(Mhyd, Polytropic, plot=False, nmore=5):
    '''
    Compute the hydrostatic moass profile and its uncertainties from an existing polytropic reconstruction

    :param Mhyd: :class:`hydromass.mhyd.Mhyd` object including the polytropic reconstruction results
    :type Mhyd: :class:`hydromass.mhyd.Mhyd`
    :param Polytropic: Polytropic model defined using the :class:`hydromass.polytropic.Polytropic` class
    :type Polytropic: :class:`hydromass.polytropic.Polytropic`
    :param plot: If True, produce a plot of the hydrostatic mass and gas mass. Defaults to False
    :type plot: bool
    :param nmore: Number of subgrid values to compute the fine binning. Each input bin will be split into nmore values. Defaults to 5.
    :type nmore: int
    :return: Dictionary including the profiles of hydrostatic mass, gas mass, and gas fraction
    :rtype: dict(11xnpt)
    '''

    nsamp = len(Mhyd.samples)

    rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore=nmore)

    nvalm = len(rin_m)

    dens_m = np.sqrt(np.dot(Mhyd.Kdens_m, np.exp(Mhyd.samples.T)) / Mhyd.ccf * Mhyd.transf)

    grad_dens = np.dot(Mhyd.Kdens_grad, np.exp(Mhyd.samples.T)) / 2. / dens_m ** 2 / Mhyd.ccf * Mhyd.transf

    p3d = Polytropic.func_np(rout_m, Mhyd.samppar, dens_m, grad_dens)

    der_lnP = Polytropic.func_der(rout_m, Mhyd.samppar, dens_m, grad_dens)

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

def prof_poly_hires(Mhyd, Polytropic, nmore=5, Z=0.3):
    """
    Extract the best-fitting thermodynamic profiles and their error envelopes from an existing polytropic reconstruction run. The output thermodynamic profiles include pressure, gas density, temperature, entropy, cooling function, and radiative cooling time.

    :param Mhyd: :class:`hydromass.mhyd.Mhyd` object including the polytropic reconstruction results
    :type Mhyd: :class:`hydromass.mhyd.Mhyd`
    :param Polytropic: Polytropic model defined using the :class:`hydromass.polytropic.Polytropic` class
    :type Polytropic: :class:`hydromass.polytropic.Polytropic`
    :param nmore: Number of subgrid values to compute the fine binning. Each input bin will be split into nmore values. Defaults to 5.
    :type nmore: int
    :param Z: Metallicity respective to Solar for the calculation of the cooling function. Defaults to 0.3.
    :type Z: float
    :return: Dictionary including the thermodynamic profiles
    :rtype: dict(23xnpt)
    """

    rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore=nmore)

    vx = MyDeprojVol(rin_m / Mhyd.amin2kpc, rout_m / Mhyd.amin2kpc)

    vol_x = vx.deproj_vol().T

    dens_m = np.sqrt(np.dot(Mhyd.Kdens_m, np.exp(Mhyd.samples.T)) / Mhyd.ccf * Mhyd.transf)

    grad_dens = np.dot(Mhyd.Kdens_grad, np.exp(Mhyd.samples.T)) / 2. / dens_m ** 2 / Mhyd.ccf * Mhyd.transf

    p3d = Polytropic.func_np(rout_m, Mhyd.samppar, dens_m, grad_dens)

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


class Polytropic:
    '''
    Class defining a polytropic model. The pressure profile is defined as a function of the gas density

    .. math::

        P_{3D}(r) = P_{0} n_e^{\\Gamma}

    The polytropic index :math:`\\Gamma` is usually not constant in galaxy clusters, therefore hydromass provides two ways of accounting for the variations of :math:`\\Gamma`:

            - Radial model: this is a slight modification of the radial polytropic model proposed by Ghirardini et al. (2019). Here the polytropic index is modified by a term that depends both on radius and on the local density gradient. See :func:`hydromass.polytropic.func_poly_rad_pm`
            - Density dependent model: a new model in which the polytropic index depends only on the local gas density. We introduce a functional form for the P(n) relation of Ghirardini et al. (2019), which provides a good description of the data for the X-COP sample. See :func:`hydromass.polytropic.func_poly_dens_pm`

    The hydrostatic mass profile can then be worked out directly from the gas density and the model polytropic index.

    :param model: Choice of polytropic model. Current implementations are 'GammaR' (radial) and 'GammaN' (density dependent).
    :type model: str
    :param start: Central values for the polytropic model parameters. If None, the best-fitting values for the X-COP sample are used. Defaults to None.
    :type start: numpy.ndarray
    :param sd: Standard deviations of the Normal priors on the polytropic model parameters. If None, "sensible" priors based on X-COP data are used. Defaults to None.
    :type sd: numpy.ndarray
    :param limits: Upper and lower boundaries of the polytropic model parameters. If None, "sensible" boundaries are used. Defaults to None.
    :type limits: numpy.ndarray
    :param fix: Array of boolean values setting which parameters are free while fitting (False) and which are fixed (True). If None, default choices are set. Defaults to None.
    :type fix: numpy.ndarray
    :param redshift: Source redshift. If None, no self-similar scaling is applied to the density, i.e. E(z)=1. Defaults to None
    :type redshift: float
    :param cosmo: Cosmology definition in astropy format. If None, Planck 2015 cosmology is assumed. Defaults to None.
    :type cosmo: astropy.cosmology
    '''

    def __init__(self, model, start=None, sd=None, limits=None, fix=None, redshift=None, cosmo=None):


        if model == 'GammaR':

            self.npar = 4

            self.parnames = ['p0', 'Gamma0', 'GammaR', 'Rs']

            if start is None:
                self.start = [5., 0.25, 0.14, 200.] # polytropic model with Gamma(R) from Ghirardini+19

            else:
                try:
                    assert (len(start) == self.npar)
                except AssertionError:
                    print('Number of starting parameters does not match function.')
                    return

                self.start = start

            if sd is None:
                self.sd = [5., 0.5, 0.5, 100.]

            else:

                try:
                    assert (len(sd) == self.npar)
                except AssertionError:
                    print('Shape of sd does not match function.')
                    return

                self.sd = sd


            if limits is None:

                limits = np.empty((self.npar, 2))

                limits[0] = [0.1, 50.]

                limits[1] = [-0.5, 2.]

                limits[2] = [-0.5, 2.]

                limits[3] = [20., 600.]

            else:

                try:
                    assert (limits.shape == (self.npar,2))
                except AssertionError:
                    print('Shape of limits does not match function.')
                    return

            if fix is None:

                self.fix = [False, False, False, False]

            else:

                try:
                    assert (len(fix) == self.npar)
                except AssertionError:
                    print('Shape of fix vectory does not match function.')
                    return

                self.fix = fix

            self.limits = limits

            self.func_pm = func_poly_rad_pm

            self.func_np = func_poly_rad_np

            self.model = model

        if model == 'GammaN':

            self.npar = 5

            self.parnames = ['p0', 'logn0', 'trans', 'Gamma0', 'scale']

            if redshift is not None:

                if cosmo is None:

                    from astropy.cosmology import Planck15 as cosmo

                eofz = cosmo.efunc(redshift)

                logn0 = -6.3299699 + 2. * np.log(eofz)

            else:

                logn0 = -6.3299699

            if start is None:
                self.start = [ 0.02, logn0 ,  1.02003952,  0.99, -0.152564  ] #best-fit parameters to X-COP Gamma(n), for P500 ~ 3e-3

            else:
                try:
                    assert (len(start) == self.npar)
                except AssertionError:
                    print('Number of starting parameters does not match function.')
                    return

                self.start = start

            if sd is None:
                self.sd = [0.06, 0.5, 0.1, 0.1, 0.05]

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

                limits[1] = [-10, -2.]

                limits[2] = [0.2, 5.]

                limits[3] = [1., 3.]

                limits[4] = [-0.5, 0.3]

            else:

                try:
                    assert (limits.shape == (self.npar,2))
                except AssertionError:
                    print('Shape of limits does not match function.')
                    return

            if fix is None:

                self.fix = [False, True, True, True, True]

            else:

                try:
                    assert (len(fix) == self.npar)
                except AssertionError:
                    print('Shape of fix vectory does not match function.')
                    return

                self.fix = fix

            self.limits = limits

            self.func_pm = func_poly_dens_pm

            self.func_np = func_poly_dens_np

            self. func_der = gradP_gamman

            self.model = model



def Run_Polytropic_PyMC3(Mhyd, Polytropic, bkglim=None,nmcmc=1000,fit_bkg=False,back=None,
                   samplefile=None,nrc=None,nbetas=6,min_beta=0.6, nmore=5,
                   tune=500, find_map=True):
    """
    Fit the data within a :class:`hydromass.mhyd.Mhyd` object with an effective polytropic index model. Here the pressure and temperature profiles are defined as a function of the gas density according to an effective polytropic relation,

    .. math::

        P_{3D}(r) = P_0 n_e(r)^{\\Gamma}

    In most practical cases :math:`\\Gamma` is not a constant across the entire radial range. The code provides two ways of modeling the variations of :math:`\\Gamma` across a cluster: either as a function of radius or as a function of gas density. The relation between density and pressure is a very universal one (see Ghirardini et al. 2019), thus this physically motivated model allows to determine the 3D profiles and hydrostatic mass of clusters even when the available data quality is quite low, e.g. with a single temperature measurement or a low-quality SZ profile. The variation of the pressure and temperature is then modeled directly from the gas density, which is usually much easier to determine.

    :param Mhyd: A :class:`hydromass.mhyd.Mhyd` object including the loaded data and initial setup (mandatory input)
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param Polytropic: Polytropic model defined using the :class:`hydromass.polytropic.Polytropic` class
    :type Polytropic: :class:`hydromass.polytropic.Polytropic`
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

        K = calc_sb_operator_psf(rad, sourcereg, pars, area, exposure, psfmat) # transformation to surface brightness

    # Set up initial values
    if np.isnan(sb[0]) or sb[0] <= 0:
        testval = -10.
    else:
        testval = np.log(sb[0] / npt)
    if np.isnan(back) or back == 0:
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

    hydro_model = pm.Model()

    cf = Mhyd.ccf

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

        for i in range(Polytropic.npar):

            name = Polytropic.parnames[i]

            if not Polytropic.fix[i]:

                lim = Polytropic.limits[i]

                if name == 'p0':

                    tpar = pm.TruncatedNormal(name, mu=np.log(Polytropic.start[i]), sigma=Polytropic.sd[i] / Polytropic.start[i],
                                                lower=np.log(lim[0]), upper=np.log(lim[1])) #log-normal prior on normalization

                    modpar = np.exp(tpar)

                else:

                    modpar = pm.TruncatedNormal(name, mu=Polytropic.start[i], sigma=Polytropic.sd[i],
                                                lower=lim[0], upper=lim[1]) #Gaussian prior on other parameters
            else:

                dummy = pm.Normal('dummy'+name, mu=0., sigma=1.)

                dummy_param = 0 * dummy + Polytropic.start[i]

                modpar = pm.Deterministic(name, dummy_param)

            allpmod.append(modpar)

        pmod = pm.math.stack(allpmod, axis=0)

        dens_m = pm.math.sqrt(pm.math.dot(Kdens_m, al) / cf * transf)  # electron density in cm-3

        dens_grad = pm.math.dot(Kdens_grad, al) / 2. / dens_m ** 2 / cf * transf # d(logn)/d(log r)

        p3d = Polytropic.func_pm(rout_m, pmod, dens_m, dens_grad)

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

                    trace = pm.sample(nmcmc, initvals=start, init='ADVI',  tune=tune, return_inferencedata=True, target_accept=0.9)

                else:

                    trace = pmjax.sample_numpyro_nuts(nmcmc, initvals=start, init='ADVI',  tune=tune,
                                                      return_inferencedata=True, target_accept=0.9)

            else:

                if not isjax:

                    trace = pm.sample(nmcmc, tune=tune, init='ADVI', target_accept=0.9)

                else:

                    trace = pmjax.sample_numpyro_nuts(nmcmc, tune=tune, init='ADVI', target_accept=0.9)

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

        else:
            Ksb = calc_sb_operator(rad, sourcereg, pars, withbkg=False)

            allsb = np.dot(Ksb, np.exp(samples.T))

        pmc = np.median(allsb, axis=1)
        pmcl = np.percentile(allsb, 50. - 68.3 / 2., axis=1)
        pmch = np.percentile(allsb, 50. + 68.3 / 2., axis=1)
        Mhyd.sb = pmc
        Mhyd.sb_lo = pmcl
        Mhyd.sb_hi = pmch

        Mhyd.nrc = nrc
        Mhyd.nbetas = nbetas
        Mhyd.min_beta = min_beta
        Mhyd.nmore = nmore
        Mhyd.pardens = pardens
        Mhyd.fit_bkg = fit_bkg

        alldens = np.sqrt(np.dot(Kdens, np.exp(samples.T)) / cf * transf)
        pmc = np.median(alldens, axis=1)
        pmcl = np.percentile(alldens, 50. - 68.3 / 2., axis=1)
        pmch = np.percentile(alldens, 50. + 68.3 / 2., axis=1)
        Mhyd.dens = pmc
        Mhyd.dens_lo = pmcl
        Mhyd.dens_hi = pmch

        samppar = np.empty((len(samples), Polytropic.npar))
        for i in range(Polytropic.npar):

            name = Polytropic.parnames[i]

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
        Mhyd.Kdens_grad = Kdens_grad

        if Mhyd.spec_data is not None:
            kt_mod = kt_poly_from_samples(Mhyd, Polytropic, nmore=nmore)
            Mhyd.ktmod = kt_mod['TSPEC']
            Mhyd.ktmod_lo = kt_mod['TSPEC_LO']
            Mhyd.ktmod_hi = kt_mod['TSPEC_HI']
            Mhyd.kt3d = kt_mod['T3D']
            Mhyd.kt3d_lo = kt_mod['T3D_LO']
            Mhyd.kt3d_hi = kt_mod['T3D_HI']

        if Mhyd.sz_data is not None:
            pmed, plo, phi = P_poly_from_samples(Mhyd, Polytropic, nmore=nmore)
            Mhyd.pmod = pmed
            Mhyd.pmod_lo = plo
            Mhyd.pmod_hi = phi
