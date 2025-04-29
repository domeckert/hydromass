import numpy as np
import matplotlib.pyplot as plt
from .deproject import *
from .constants import *
from .wl import *
from .pnt import *
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import copy
import pyproffit

def get_coolfunc(Z):
    """
    Retrieve the cooling function data from the internal data archive and return the cooling function profile for a given input metallicity

    :param Z: Metallicity (relative to Anders & Grevesse; defaults to 0.3)
    :type Z: float
    :return: Cooling function profile and temperature grid
    :rtype: numpy.ndarray
    """
    Z_grid = np.array([0., 0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.35, 0.40, 0.48, 0.58, 0.70, 0.85, 1.0])

    file_cf = get_data_file_path('coolfunc_table.fits')

    fcf = fits.open(file_cf)

    ind = np.searchsorted(Z_grid, Z, side='left')

    if Z_grid[ind] < Z:

        Zlow = Z_grid[ind]

        Zhigh = Z_grid[ind + 1]

    elif Z_grid[ind] > Z:

        Zlow = Z_grid[ind - 1]

        Zhigh = Z_grid[ind]

    else:

        Zlow = Z

        Zhigh = Z

    if Zlow != Zhigh:

        hdulow = fcf['COOLFUNC_Z%1.2lf' % (Zlow)]

        lambdalow = hdulow.data['LAMBDA']

        ktgrid = hdulow.data['KT']

        hduhigh = fcf['COOLFUNC_Z%1.2lf' % (Zhigh)]

        lambdahigh = hduhigh.data['LAMBDA']

        lambda_interp = lambdalow + (Z - Zlow) * (lambdahigh - lambdalow) / (Zhigh - Zlow)

    else:

        thdu = fcf['COOLFUNC_Z%1.2lf' % (Z)]

        lambda_interp = thdu.data['LAMBDA']

        ktgrid = thdu.data['KT']

    return lambda_interp, ktgrid

def cumsum_mat(nval):
    """

    Function to create a matrix that flips a vector, makes a cumulative sum and adds a 0 as the first element
    Then dot(mat, vector) returns the Riemann integral as a cumulative sum

    :param nval: Vector size
    :type nval: int
    :return: Cumulative sum operator
    :type nval: numpy.ndarray
    """
    onemat = np.ones((nval - 1, nval - 1))

    triu = np.triu(onemat)

    flipped_triu = np.flipud(triu)

    zeromat = np.zeros((nval, nval))

    zeromat[1:, 1:] = flipped_triu

    totmat = np.flipud(zeromat)

    return  totmat


def rads_more(Mhyd, nmore=5, extend=False):
    """

    Return grid of (in, out) radii from X-ray, SZ data or both. Concatenates radii if necessary, then computes a grid of radii.
    Returns the output arrays and the indices corresponding to the input X-ray and/or SZ radii.

    :param Mhyd: A :class:`hydromass.mhyd.Mhyd` object containing loaded X-ray and/or SZ loaded data.
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param nmore: Number of subgrid values to compute the fine binning. Each input bin will be split into nmore values. Defaults to 5.
    :type nmore: int
    :return:
        - rin, rout: the inner and outer radii of the fine grid
        - index_x, index_sz: lists of indices corresponding to the position of the input values in the grid
        - sum_mat: matrix containing the number of values in each subgrid bin
        - ntm: total number of grid points
    """
    if Mhyd.spec_data is not None and Mhyd.sz_data is None:

        rout_joint = Mhyd.spec_data.rout_x

        rref_joint = Mhyd.spec_data.rref_x

    elif Mhyd.spec_data is None and Mhyd.sz_data is not None:

        rout_joint = Mhyd.sz_data.rout_sz

        rref_joint = Mhyd.sz_data.rref_sz

    elif Mhyd.spec_data is not None and Mhyd.sz_data is not None:

        rout_joint = np.sort(np.append(Mhyd.spec_data.rout_x, Mhyd.sz_data.rout_sz))

        rref_joint = np.sort(np.append(Mhyd.spec_data.rref_x, Mhyd.sz_data.rref_sz))

    else:

        print('No loaded data found in input hydromass.Mhyd object, nothing to do')

        return

    rin_joint = np.roll(rout_joint, 1)

    rin_joint[0] = 0.

    njoint = len(rref_joint)

    tot_joint = np.sort(np.append(rin_joint, rref_joint))

    ntotjoint = len(tot_joint)

    ntm = int((ntotjoint - 0.5) * nmore)

    rout_more = np.empty(ntm)

    for i in range(ntotjoint - 1):

        rout_more[i * nmore:(i + 1) * nmore] = np.linspace(tot_joint[i], tot_joint[i + 1], nmore + 1)[1:]

    rout_more[(ntotjoint - 1) * nmore:] = np.linspace(rref_joint[njoint - 1], rout_joint[njoint - 1], int(nmore / 2.) + 1)[1:]

    # Move the outer boundary to the edge of the SB profile if it is farther out
    sbprof = Mhyd.sbprof

    rmax_sb = np.max(sbprof.bins) * Mhyd.amin2kpc

    if rmax_sb > np.max(rout_more):
        nvm = len(rout_more)

        dx_out = rout_more[nvm - 1] - rout_more[nvm - 2]

        rout_2add = np.arange(np.max(rout_more), rmax_sb, dx_out)

        rout_2add = np.append(rout_2add[1:], rmax_sb)

        rout_more = np.append(rout_more, rout_2add)

    rin_more = np.roll(rout_more, 1)

    rin_more[0] = 0

    index_x, index_sz = None, None

    if Mhyd.spec_data is not None:

        index_x = np.where(np.in1d(rout_more, Mhyd.spec_data.rref_x))

    if Mhyd.sz_data is not None:

        index_sz = np.where(np.in1d(rout_more, Mhyd.sz_data.rref_sz))

    sum_mat = None

    if Mhyd.spec_data is not None:

        ntot = len(rout_more)

        nx = len(Mhyd.spec_data.rref_x)

        if not extend:

            sum_mat = np.zeros((nx, ntot))

        else:

            if Mhyd.spec_data.rout_x[nx-1] < np.max(rin_more):
                sum_mat = np.zeros((nx+1, ntot))

            else:
                sum_mat = np.zeros((nx, ntot))

        for i in range(nx):

            ix = np.where(np.logical_and(rin_more < Mhyd.spec_data.rout_x[i], rin_more >= Mhyd.spec_data.rin_x[i]))

            nval = len(ix[0])

            sum_mat[i, :][ix] = 1. / nval

        if extend:

            if Mhyd.spec_data.rout_x[nx-1] < np.max(rin_more):

                ix = np.where(rin_more >= Mhyd.spec_data.rout_x[nx-1])

                nval = len(ix[0])

                sum_mat[nx, :][ix] = 1. / nval

    return rin_more, rout_more, index_x, index_sz, sum_mat, ntm

def gnfw_p0(x,pars):
    '''
    Generalized NFW function to estimate the pressure at the outer boundary, P0

    .. math::

        P_{gNFW}(r) = \\frac{P_0} {(r/r_s)^\\gamma (1+(r/rs)^\\alpha)^{(\\beta-\\gamma)/\\alpha}}

    :param x: Radius
    :type x: numpy.ndarray
    :param pars: Array containing the five parameters (P0, rs, alpha, beta, and gamma) of the gNFW function
    :type pars: numpy.ndarray
    :return: Model pressure
    :rtype: numpy.ndarray
    '''
    P0=pars[0]
    rs=pars[1]
    alpha=pars[2]
    beta=pars[3]
    gamma=pars[4]
    t1=np.power(x/rs,gamma)
    t2=np.power(1.+np.power(x/rs,alpha),(beta-gamma)/alpha)
    return P0/t1/t2


def estimate_P0(Mhyd, dens='sb', outfile=None):
    '''
    Provide an estimate of the pressure at the outer boundary by fitting a rough gNFW profile to the data. The value of :math:`P_0` is the integration constant that enters when we integrate the hydrostatic equilibrium equation to predict the pressure profile,

    .. math::

        P(r) = P_0 + \\int_0^r \\frac{GM(<r^{\\prime})} {r^{\\prime 2}} dr^{\\prime}

    A rough electron density profile is estimated by fitting a double beta model profile to the surface brightness, and temperature deprojection is neglected. The resulting pressure profile is fitted with a gNFW profile using the scipy.minimize function and the best-fit function is used to extrapolate the pressure to the outer boundary to provide a rough estimate of :math:`P_0`.

    :param Mhyd: A :class:`hydromass.mhyd.Mhyd` object containing the loaded data
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param dens: Set whether we will deproject the surface brightness profile (dens='sb') or the normalization of the spectrum (dens='norm'). Defaults to 'sb'
    :type dens: str
    :return: Estimated value of :math:`P_0`
    :rtype: float
    '''
    spec_data = Mhyd.spec_data

    if dens == 'sb' or spec_data.norm is None:
        sbprof = copy.copy(Mhyd.sbprof)

        modbeta = pyproffit.Model(pyproffit.DoubleBeta)

        fitbeta = pyproffit.Fitter(model=modbeta, profile=sbprof, bkg=-12., beta=0.7, rc1=0.5, rc2=2., norm=-2, ratio=2)

        fitbeta.minuit.fixed['bkg'] = True

        fitbeta.Migrad()

        rc1 = np.abs(fitbeta.params['rc1']) * Mhyd.amin2kpc  # kpc
        rc2 = np.abs(fitbeta.params['rc2']) * Mhyd.amin2kpc

        beta = fitbeta.params['beta']
        norm = fitbeta.params['norm']
        ratio = fitbeta.params['ratio']

        cfact1 = gamma(3 * beta) / gamma(3 * beta - 0.5) / np.sqrt(np.pi) / rc1
        cfact2 = gamma(3 * beta) / gamma(3 * beta - 0.5) / np.sqrt(np.pi) / rc2

        rfit = sbprof.bins * Mhyd.amin2kpc

        t1 = (1. + np.power(rfit / rc1, 2)) ** (-3. * beta) * cfact1
        t2 = (1. + np.power(rfit / rc2, 2)) ** (-3. * beta) * cfact2

        dens_prof = np.sqrt(10 ** norm * (t1 + ratio * t2) / Mhyd.ccf * Mhyd.transf)

        ne_interp = np.interp(spec_data.rref_x_am, sbprof.bins, dens_prof)

        p_interp = ne_interp * spec_data.temp_x
        ep_interp = ne_interp * spec_data.errt_x

    else:
        sbprof = copy.copy(Mhyd.sbprof)

        dat = sbprof.data
        cra = sbprof.cra
        cdec = sbprof.cdec

        prof_n = pyproffit.Profile(dat, center_choice='custom_fk5', center_ra=cra, center_dec=cdec,
                                   binsize=3., maxrad=11.)

        prof_n.bins = spec_data.rref_x_am
        prof_n.ebins = (spec_data.rout_x_am - spec_data.rin_x_am) / 2.
        prof_n.profile = spec_data.norm
        prof_n.eprof = spec_data.norm_lo
        prof_n.nbin = len(spec_data.rref_x_am)

        depr = pyproffit.Deproject(z=Mhyd.redshift, cf=1., profile=prof_n)

        depr.OnionPeeling()

        p_interp = depr.dens * spec_data.temp_x
        ep_interp = depr.dens * spec_data.errt_x

    pars_press = np.array([3.28, 1200., 1.33, 4.72, 0.59])

    def chi2_gnfw(pars):
        pars_press[0] = 10 ** pars[0]
        pars_press[1] = pars[1]
        mm = gnfw_p0(spec_data.rref_x, pars_press)
        chi2 = np.sum((p_interp - mm) ** 2 / ep_interp ** 2)
        return chi2

    bnds = ((None, None), (0, None))

    res = minimize(chi2_gnfw, np.array([-4, 1200.]), method='Nelder-Mead', bounds=bnds)

    maxrad = np.max(sbprof.bins * Mhyd.amin2kpc)

    if Mhyd.sz_data is not None:

        rmaxsz = np.max(Mhyd.rout_sz)

        if rmaxsz > maxrad:

            maxrad = rmaxsz

    pars_press[0] = 10 ** res['x'][0]
    pars_press[1] = res['x'][1]

    p0 = gnfw_p0(maxrad, pars_press)

    if outfile is not None:

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

        plt.xscale('log')

        plt.yscale('log')

        plt.errorbar(spec_data.rref_x, p_interp, yerr=ep_interp, fmt='o', markersize=10, label='P data')

        xp = np.logspace(np.min(spec_data.rref_x), maxrad, 100)

        yp = gnfw_p0(xp, pars_press)

        plt.plot(xp, yp, label='P model')

        plt.plot(maxrad, p0, 's', markersize=20, label='P0')

        plt.savefig(outfile)

    return p0


def estimate_T0(Mhyd):
    '''
    Provide an estimate of the temperature at the outer boundary by fitting a rough gNFW profile to the data. The value of :math:`P_0` is the integration constant that enters when we integrate the hydrostatic equilibrium equation to predict the pressure profile,

    .. math::

        P(r) = P_0 + \\int_0^r \\frac{GM(<r^{\\prime})} {r^{\\prime 2}} dr^{\\prime}

    A rough electron density profile is estimated by fitting a double beta model profile to the surface brightness, and temperature deprojection is neglected. The resulting pressure profile is fitted with a gNFW profile using the scipy.minimize function and the best-fit function is used to extrapolate the pressure to the outer boundary to provide a rough estimate of :math:`P_0`.

    :param Mhyd: A :class:`hydromass.mhyd.Mhyd` object containing the loaded data
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param dens: Set whether we will deproject the surface brightness profile (dens='sb') or the normalization of the spectrum (dens='norm'). Defaults to 'sb'
    :type dens: str
    :return: Estimated value of :math:`P_0`
    :rtype: float
    '''
    spec_data = Mhyd.spec_data

    neout = None

    sbprof = copy.copy(Mhyd.sbprof)

    maxrad = np.max(sbprof.bins * Mhyd.amin2kpc)

    if Mhyd.sz_data is not None:

        rmaxsz = np.max(Mhyd.rout_sz)

        if rmaxsz > maxrad:
            maxrad = rmaxsz

    modbeta = pyproffit.Model(pyproffit.DoubleBeta)

    fitbeta = pyproffit.Fitter(model=modbeta, profile=sbprof, bkg=-12., beta=0.7, rc1=0.5, rc2=2., norm=-2, ratio=2)

    fitbeta.minuit.fixed['bkg'] = True

    fitbeta.Migrad()

    rc1 = np.abs(fitbeta.params['rc1']) * Mhyd.amin2kpc  # kpc
    rc2 = np.abs(fitbeta.params['rc2']) * Mhyd.amin2kpc

    beta = fitbeta.params['beta']
    norm = fitbeta.params['norm']
    ratio = fitbeta.params['ratio']

    cfact1 = gamma(3 * beta) / gamma(3 * beta - 0.5) / np.sqrt(np.pi) / rc1
    cfact2 = gamma(3 * beta) / gamma(3 * beta - 0.5) / np.sqrt(np.pi) / rc2

    rfit = sbprof.bins * Mhyd.amin2kpc

    t1 = (1. + np.power(rfit / rc1, 2)) ** (-3. * beta) * cfact1
    t2 = (1. + np.power(rfit / rc2, 2)) ** (-3. * beta) * cfact2

    dens_prof = np.sqrt(10 ** norm * (t1 + ratio * t2) / Mhyd.ccf * Mhyd.transf)

    ne_interp = np.interp(spec_data.rref_x_am, sbprof.bins, dens_prof)

    p_interp = ne_interp * spec_data.temp_x
    ep_interp = ne_interp * spec_data.errt_x

    t1out = (1. + np.power(maxrad / rc1, 2)) ** (-3. * beta) * cfact1
    t2out = (1. + np.power(maxrad / rc2, 2)) ** (-3. * beta) * cfact2
    neout = np.sqrt(10 ** norm * (t1out + ratio * t2out) / np.median(Mhyd.ccf) * Mhyd.transf)

    pars_press = np.array([3.28, 1200., 1.33, 4.72, 0.59])

    def chi2_gnfw(pars):
        pars_press[0] = 10 ** pars[0]
        pars_press[1] = pars[1]
        mm = gnfw_p0(spec_data.rref_x, pars_press)
        chi2 = np.sum((p_interp - mm) ** 2 / ep_interp ** 2)
        return chi2

    bnds = ((None, None), (0, None))

    res = minimize(chi2_gnfw, np.array([-4, 1200.]), method='Nelder-Mead', bounds=bnds)

    pars_press[0] = 10 ** res['x'][0]
    pars_press[1] = res['x'][1]

    p0 = gnfw_p0(maxrad, pars_press)

    T0 = p0 / neout

    return T0

def densout_pout_from_samples(Mhyd, model, rin_m, rout_m):
    '''
    Compute the model 3D density and pressure profiles from the output NUTS sample on an arbitrary output grid

    :param Mhyd: A :class:`hydromass.mhyd.Mhyd` object containing the result of a mass model fit
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param model: A :class:`hydromass.functions.Model` object containing the definition of the mass model
    :type model: class:`hydromass.functions.Model`
    :param rin_m: A 1-D array containing the inner boundaries of the chosen bins
    :type rin_m: numpy.ndarray
    :param rout_m: A 1-D array containing the outer boundaries of the chosen bins
    :type rout_m: numpy.ndarray
    :return:
        - dens_m: A 2-D array containing the gas density profiles for all the samples
        - press_out: A 2-D array containing the total 3D pressure profile
        - pth: A 2-D array containing the thermal pressure profile. If non-thermal pressure is not modeled then this is equal to the total pressure
    '''
    samples = Mhyd.samples

    nsamp = len(samples)

    nvalm = len(rin_m)

    rref_m = (rin_m + rout_m) / 2.

    if Mhyd.cf_prof is not None:

        rad = Mhyd.sbprof.bins

        tcf = np.interp(rref_m, rad * Mhyd.amin2kpc, Mhyd.ccf)

        if np.isscalar(Mhyd.elong):

            cf_prof = np.repeat(tcf, nsamp).reshape(nvalm, nsamp)

        else:

            cf_prof = tcf

    else:

        cf_prof = Mhyd.ccf

    rref_m = (rin_m + rout_m) / 2.

    if not np.isscalar(Mhyd.elong) and not Mhyd.fit_bkg:

        dens_m = np.empty((nsamp, nvalm))

        for i in range(nsamp):

            Kdens_t = calc_density_operator_pm(rref_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.elong[i], Mhyd.amin2kpc)

            dens_m[i] = np.sqrt(np.dot(Kdens_t, np.exp(samples[i])) / cf_prof * Mhyd.transf)  # electron density in cm-3

    else:

        Kdens_m = calc_density_operator(rref_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc, withbkg=Mhyd.fit_bkg)

        dens_m = np.transpose(np.sqrt(np.dot(Kdens_m, np.exp(samples.T)) / cf_prof * Mhyd.transf))

    mass = Mhyd.mfact * model.func_np(rref_m, Mhyd.samppar, delta=model.delta) / Mhyd.mfact0

    rout_mul = np.tile(rout_m, nsamp).reshape(nsamp, nvalm)

    rin_mul = np.tile(rin_m, nsamp).reshape(nsamp, nvalm)

    rref_mul = np.tile(rref_m, nsamp).reshape(nsamp, nvalm)

    # Adding baryonic mass contribution in case of DM-only fit
    if Mhyd.dmonly:
        # Matrix containing integration volumes
        volmat = np.repeat(4. / 3. * np.pi * (rout_m ** 3 - rin_m ** 3), nsamp).reshape(nvalm, nsamp)

        # Compute Mgas profile as cumulative sum over the volume

        nhconv = cgsamu * Mhyd.mu_e * cgskpc ** 3 / Msun  # Msun/kpc^3

        ones_mat = np.ones((nvalm, nvalm))

        cs_mat = np.tril(ones_mat)

        mgas = np.dot(cs_mat, dens_m * nhconv * volmat) / 1e13 / Mhyd.mfact0

        if Mhyd.mstar is not None:

            r_mstar = Mhyd.mstar[:, 0]

            cum_mstar = Mhyd.mstar[:, 1]

            mstar_m = np.interp(rout_m, r_mstar, cum_mstar)

            mstar_mul = np.repeat(mstar_m, nsamp).reshape(nvalm, nsamp)

            mbar = mgas + mstar_mul / Mhyd.mfact0 / 1e13

        else:

            mbar = mgas

        mass = mass + mbar.T

    # Pressure gradient
    dpres = - mass / rref_mul ** 2 * dens_m * (rout_mul - rin_mul)

    press00 = np.exp(Mhyd.samplogp0)

    int_mat = cumsum_mat(nvalm)

    press_out = press00 - np.dot(int_mat, dpres.T)

    if Mhyd.pnt:

        if Mhyd.pnt_model == 'Angelinelli':

            alpha_turb = alpha_turb_np(rref_m, Mhyd.samppar, Mhyd.redshift, Mhyd.pnt_pars)

            pth = press_out * (1. - alpha_turb)

        if Mhyd.pnt_model == 'Ettori':

            log_pnt = Mhyd.pnt_pars[:,1] * np.log(dens_m.T * 1e3) + Mhyd.pnt_pars[:,0] * np.log(10)

            pth = press_out - np.exp(log_pnt)

    else:

        pth = press_out

    return dens_m.T, press_out, pth

def kt_from_samples(Mhyd, model, nmore=5):
    """
    Compute model temperature profile from a mass reconstruction run, evaluated at reference X-ray temperature radii

    :param Mhyd: A :class:`hydromass.mhyd.Mhyd` object containing the result of a mass model fit
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param model: A :class:`hydromass.functions.Model` object containing the definition of the mass model
    :type model: class:`hydromass.functions.Model`
    :return: Dictionary containing the model temperature profile and uncertainties, 3D and spectroscopic-like
    :rtype: dict(9xnval)
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

    dens_m, press_out, pth = densout_pout_from_samples(Mhyd, model, rin_m, rout_m)

    t3d = pth / dens_m

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


def P_from_samples(Mhyd, model, nmore=5, return_Y=False, nout=10):
    """
    Compute model pressure profile from an existing mass reconstruction run and evaluate it at the reference SZ radii

    :param Mhyd: A :class:`hydromass.mhyd.Mhyd` object containing the result of a mass model fit
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param model: A :class:`hydromass.functions.Model` object containing the definition of the mass model
    :type model: class:`hydromass.functions.Model`
    :return: Arrays containing the median 3D pressure profile and the 16th and 84th percentiles
    :rtype: numpy.ndarray
    """

    if Mhyd.sz_data is None:
        print('No SZ data provided')

        return

    rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore=nmore)

    dens_m, press_tot, pth = densout_pout_from_samples(Mhyd, model, rin_m, rout_m)

    print(pth.shape)

    pmt, plot, phit = np.percentile(pth, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    pmed, plo, phi = pmt[index_sz], plot[index_sz], phit[index_sz]

    nsamp = len(Mhyd.samppar)

    if return_Y == True:

        nout = 2 * Mhyd.nmore

        nsamp = len(Mhyd.samppar)

        rref_m = (rin_m + rout_m) / 2.

        rout_m_p = np.append(rout_m, np.logspace(np.log10(np.max(rout_m) * 1.1), np.log10(10000.), nout))
        rin_m_p = np.append(rin_m, rout_m_p[ntm - 1:ntm - 1 + nout])

        rref_m_p = (rin_m_p + rout_m_p) / 2.

        pth_p = np.empty((ntm + nout, nsamp))

        pth_p[:ntm, :] = pth

        slope = (np.log10(pth[ntm - 1, :]) - np.log10(pth[ntm - 10, :])) / (
                np.log10(rref_m[ntm - 1]) - np.log10(rref_m[ntm - 10]))

        P0 = pth[ntm - 1]

        routmat = np.repeat(rref_m_p[ntm:], nsamp).reshape(nout, nsamp)

        pth_p[ntm:, :] = P0 * (routmat / rref_m[ntm - 1]) ** slope

        rin_cm_p, rout_cm_p = rin_m_p * cgskpc, rout_m_p * cgskpc

        deproj_p = MyDeprojVol(rin_cm_p, rout_cm_p)  # r from kpc to cm

        proj_vol_p = deproj_p.deproj_vol().T

        area_proj_p = np.pi * (-(rin_cm_p) ** 2 + (rout_cm_p) ** 2)

        integ_p = np.dot(proj_vol_p, pth_p) / np.tile(area_proj_p[:, np.newaxis], (1, nsamp))

        y_num = y_prefactor * integ_p  # prefactor in cm2/keV

        #print("profile_values.shape:", y_num.shape)
        #print("r_values.shape:", ((rin_cm_p + rout_cm_p)/2).shape)
        #print("index_sz:", index_sz)
        #print("elongation.shape:", Mhyd.elong.shape)

        ysz = elongation_correction_np(y_num, (rin_cm_p + rout_cm_p)/2, index_sz[0], Mhyd.elong)

        if Mhyd.sz_data.psfmat is not None:
            ysz = np.dot(Mhyd.sz_data.psfmat, ysz)

        pmed, plo, phi = np.percentile(ysz, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    return pmed, plo, phi


def g_from_samples(Mhyd, model, n_draw=None, random_state=None):
    """
    Computes the tangential shear g+ from an existing mass reconstruction run, accounting for los elongation.

    :param Mhyd: A :class:`hydromass.mhyd.Mhyd` object containing the result of a mass model fit
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param model: A :class:`hydromass.functions.Model` object containing the definition of the mass model
    :type model: class:`hydromass.functions.Model`
    :param n_draw: int, optional, number of random samples to draw from pmod.
                   If None, all samples are used.
    :param random_state: int or np.random.Generator, optional, seed for reproducibility.
    :return:
        gplus_all: 2D array of tangential shear, shape (M, n_draw), where M = len(rm), n_draw = selected parameter sets
        rm: array-like, mean radii for the numerical integration
        ev: array-like, indices for the evaluation of the mass profile
    """
    WLdata = Mhyd.wl_data

    pmod = Mhyd.samppar

    elong = Mhyd.elong

    radplus, rm, ev = get_radplus(WLdata.radii_wl)
    
    # Ensure pmod is 2D for consistency
    pmod = np.atleast_2d(pmod)  # Converts 1D array to 2D if needed
    n_samples = pmod.shape[0]
    
    # Select samples if n_draw is specified
    if n_draw is not None and n_draw < n_samples:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(n_samples, n_draw, replace=False)
        pmod = pmod[indices]
        n_samples = pmod.shape[0]
    
    # Initialize the result array
    gplus_all = np.zeros((len(rm), n_samples))
    
    # Loop over all parameter sets
    for i in tqdm(range(n_samples)):
        rho_out = model.rho_np(radplus, *pmod[i]) * WLdata.rho_crit
        sig = rho_to_sigma_np(radplus, rho_out)
        _, dsigma = dsigma_trap_np(sig, radplus)
        gplus = get_shear(sig, dsigma, WLdata.msigmacrit, WLdata.fl)
        
        # Extract values at the evaluation radii
        gplus_all[:, i] = gplus

    if np.isscalar(elong) and elong == 1:
        gplus_elong = gplus_all[np.arange(len(rm)-2)+1]
    else:
        elong = elong[indices]
        gplus_elong = elongation_correction_np(gplus_all, rm, np.arange(len(rm)-2)+1, elong)    
    return gplus_elong, rm[np.arange(len(rm)-2)+1], ev


def mass_from_samples(Mhyd, model, rin=None, rout=None, npt=200, plot=False):
    """
    Compute the median and percentile mass profile, gas mass and gas fraction from an existing mass reconstruction run

    :param Mhyd: A :class:`hydromass.mhyd.Mhyd` object containing the result of a mass model fit
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param model: A :class:`hydromass.functions.Model` object containing the definition of the mass model
    :type model: class:`hydromass.functions.Model`
    :param rin: Minimum radius of the output profiles
    :type rin: float
    :param rout: Maximum radius of the output profiles
    :type rout: float
    :param npt: Number of radial points in the output profiles. Defaults to 200
    :type npt: int
    :param plot: Plot the mass and gas mass profiles and return a matplotlib figure. Defaults to False.
    :type plot: bool
    :return: Dictionary containing the median mass [in M_sun], Lower 1-sigma percentile, Upper 1-sigma percentile, Median Mgas, Lower, Upper, Median Fgas, Lower, Upper
    :rtype: dict(16xnpt)
    """

    if not Mhyd.wlonly:
        rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore=Mhyd.nmore)

        if rin is None:
            rin = np.min(rin_m)

            if rin == 0:
                rin = 1.

        if rout is None:
            rout = np.max(rout_m)

    else:

        if rin is None:
            rin = 10.

        if rout is None:
            rout = 3000.

    bins = np.logspace(np.log10(rin), np.log10(rout), npt + 1)

    if rin == 1.:
        bins[0] = 0.

    rin_m = bins[:npt]

    rout_m = bins[1:]

    rref_m = (rin_m + rout_m) / 2.

    mass = Mhyd.mfact * model.func_np(rout_m, Mhyd.samppar, model.delta) * 1e13

    nsamp = len(Mhyd.samppar)

    nvalm = len(rin_m)

    if not Mhyd.wlonly:

        if Mhyd.cf_prof is not None:

            rref_m = (rin_m + rout_m) / 2.

            rad = Mhyd.sbprof.bins

            tcf = np.interp(rref_m, rad * Mhyd.amin2kpc, Mhyd.ccf)

            cf_prof = np.repeat(tcf, nsamp).reshape(nvalm, nsamp)

        else:

            cf_prof = Mhyd.ccf

        if Mhyd.fit_bkg:

            Kdens_m = calc_density_operator(rref_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc)

        else:

            Kdens_m = calc_density_operator(rref_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc, withbkg=False)

        alldens = np.sqrt(np.dot(Kdens_m, np.exp(Mhyd.samples.T)) / cf_prof * Mhyd.transf)

        # Matrix containing integration volumes
        volmat = np.repeat(4. / 3. * np.pi * (rout_m ** 3 - rin_m ** 3), nsamp).reshape(nvalm, nsamp)

        # Compute Mgas profile as cumulative sum over the volume

        nhconv = cgsamu * Mhyd.mu_e * cgskpc ** 3 / Msun  # Msun/kpc^3

        ones_mat = np.ones((nvalm, nvalm))

        cs_mat = np.tril(ones_mat)

        mgas = np.dot(cs_mat, alldens * nhconv * volmat)

        mg, mgl, mgh = np.percentile(mgas, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

        if Mhyd.mstar is not None:

            r_mstar = Mhyd.mstar[:, 0]

            cum_mstar = Mhyd.mstar[:, 1]

            mstar_m = np.interp(rout_m, r_mstar, cum_mstar)

        else:

            mstar_m = np.zeros(nvalm)

        if Mhyd.dmonly:

            mtot = mass + mgas.T + mstar_m.T

            fgas = mgas / mtot.T

        else:

            fgas = mgas / mass.T

        fg, fgl, fgh = np.percentile(fgas, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

        mmed, mlo, mhi = np.percentile(mass, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=0)

        if Mhyd.dmonly:

            mtotm, mtotlo, mtothi = np.percentile(mtot, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=0)

        else:

            mtotm, mtotlo, mtothi = mmed, mlo, mhi

        dict = {
            "R_IN": rin_m,
            "R_OUT": rout_m,
            "R_REF": rref_m,
            "MASS": mtotm,
            "MASS_LO": mtotlo,
            "MASS_HI": mtothi,
            "M_DM": mmed,
            "M_DM_LO": mlo,
            "M_DM_HI": mhi,
            "MGAS": mg,
            "MGAS_LO": mgl,
            "MGAS_HI": mgh,
            "FGAS": fg,
            "FGAS_LO": fgl,
            "FGAS_HI": fgh,
            "M_STAR": mstar_m
        }

    else:

        mmed, mlo, mhi = np.percentile(mass, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=0)

        dict = {
            "R_IN": rin_m,
            "R_OUT": rout_m,
            "R_REF": rref_m,
            "MASS": mmed,
            "MASS_LO": mlo,
            "MASS_HI": mhi,
            "M_DM": mmed,
            "M_DM_LO": mlo,
            "M_DM_HI": mhi,
            "MGAS": np.zeros(nvalm),
            "MGAS_LO": np.zeros(nvalm),
            "MGAS_HI": np.zeros(nvalm),
            "FGAS": np.zeros(nvalm),
            "FGAS_LO": np.zeros(nvalm),
            "FGAS_HI": np.zeros(nvalm),
            "M_STAR": np.zeros(nvalm)
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

        if not Mhyd.wlonly:
            plt.plot(rout_m, mg, color='blue', label='$M_{gas}$')

            plt.fill_between(rout_m, mgl, mgh, color='blue', alpha=0.4)

        plt.plot(rout_m, mmed, color='red', label='$M_{tot}$')

        plt.fill_between(rout_m, mlo, mhi, color='red', alpha=0.4)

        if Mhyd.mstar is not None:

            plt.plot(rout_m, mstar_m, color='green', label='$M_{\star}$')

        plt.xlabel('Radius [kpc]', fontsize=40)

        plt.ylabel('$M(<R) [M_\odot]$', fontsize=40)

        plt.legend(fontsize = 22)

        return dict, fig

    else:

        return dict


def prof_hires(Mhyd, model, rin=None, npt=200, Z=0.3):
    """
    Compute best-fitting thermodynamic profiles and error envelopes from an existing mass reconstruction run

    :param Mhyd: A :class:`hydromass.mhyd.Mhyd` object containing the result of a mass model fit
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param model: A :class:`hydromass.functions.Model` object containing the definition of the mass model
    :type model: class:`hydromass.functions.Model`
    :param rin: Minimum radius of the output profiles
    :type rin: float
    :param npt: Number of radial points in the output profiles. Defaults to 200
    :type npt: int
    :param Z: Gas metallicity for cooling function calculation. Defaults to 0.3
    :type Z: float
    :return: Dictionary containing the median profiles and 1-sigma percentiles of temperature, pressure, gas density, entropy, and cooling time
    :rtype: dict(30xnpt)
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

    dens_m, p3d, pth = densout_pout_from_samples(Mhyd, model, rin_m, rout_m)

    t3d = pth / dens_m

    # Mazzotta weights
    ei = dens_m ** 2 * t3d ** (-0.75)

    # Temperature projection
    flux = np.dot(vol_x, ei)

    tproj = np.dot(vol_x, t3d * ei) / flux

    K3d = t3d * dens_m ** (- 2. / 3.)

    mptot, mptotl, mptoth = np.percentile(p3d, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    mp, mpl, mph = np.percentile(pth, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    mt3d, mt3dl, mt3dh = np.percentile(t3d, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    mtp, mtpl, mtph = np.percentile(tproj, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    mne, mnel, mneh = np.percentile(dens_m, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    mK, mKl, mKh = np.percentile(K3d, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    coolfunc, ktgrid = get_coolfunc(Z)

    lambda3d = np.interp(t3d, ktgrid, coolfunc)

    tcool = 3. / 2. * dens_m * (1. + 1. / Mhyd.nhc) * t3d * kev2erg / (
                lambda3d * dens_m ** 2 / Mhyd.nhc) / year

    mtc, mtcl, mtch = np.percentile(tcool, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    mcf, mcfl, mcfh = np.percentile(lambda3d, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    if Mhyd.pnt:

        pnt_all = p3d - pth

        alpha_all = pnt_all / p3d

        sigma_all = np.nan_to_num(np.sqrt(pnt_all * kev2erg * 3. / Mhyd.mup / cgsamu / dens_m)) / 1e5 # km/s

        mpnt, mpntl, mpnth = np.percentile(pnt_all, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

        msigma, sigmal, sigmah = np.percentile(sigma_all, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

        malpha, alphal, alphah = np.percentile(alpha_all, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    else:

        mpnt, mpntl, mpnth = np.zeros(len(mptot)), np.zeros(len(mptot)), np.zeros(len(mptot))

        msigma, sigmal, sigmah = np.zeros(len(mptot)), np.zeros(len(mptot)), np.zeros(len(mptot))

        malpha, alphal, alphah = np.zeros(len(mptot)), np.zeros(len(mptot)), np.zeros(len(mptot))

    dict = {
        "R_IN": rin_m,
        "R_OUT": rout_m,
        "R_REF": rref_m,
        "P_TOT": mptot,
        "P_TOT_LO": mptotl,
        "P_TOT_HI": mptoth,
        "P_TH": mp,
        "P_TH_LO": mpl,
        "P_TH_HI": mph,
        #"Y": yszm,
        #"Y_LO": yszl,
        #"Y_HI": yszh,
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
        "P_NT": mpnt,
        "P_NT_LO": mpntl,
        "P_NT_HI": mpnth,
        "SIGMA": msigma,
        "SIGMA_LO": sigmal,
        "SIGMA_HI": sigmah,
        "ALPHA_NT": malpha,
        "ALPHA_NT_LO": alphal,
        "ALPHA_NT_HI": alphah,
        "T_COOL": mtc,
        "T_COOL_LO": mtcl,
        "T_COOL_HI": mtch,
        "LAMBDA": mcf,
        "LAMBDA_LO": mcfl,
        "LAMBDA_HI": mcfh
    }

    return dict

def mgas_pm(rin_m, rout_m, dens):
    '''
    Theano function to compute the gas mass

    :param rin_m: 1-D array containing the inner edges of radial bins
    :type rin_m: numpy.ndarray
    :param rout_m: 1-D array containing the outer edges of radial bins
    :type rout_m: numpy.ndarray
    :param dens: Theano tensor including the density profile evaluated at the chosen radial bins
    :type dens: theano.tensor
    :return: Cumulative gas mass profile
    :rtype: theano.tensor
    '''

    # Integration volumes
    volint = 4. /3. * np.pi * (rout_m ** 3 - rin_m ** 3)

    nvalm = len(rin_m)

    ones_mat = np.ones((nvalm, nvalm))

    cs_mat = np.tril(ones_mat)

    mgas = pm.math.dot(cs_mat, dens * volint) / 1e13

    return mgas


def PlotMgas(Mhyd, plot=False, outfile=None, nmore=5):
    """
    Compute Mgas profile and error envelope from mass reconstruction run

    :param Mhyd: A :class:`hydromass.mhyd.Mhyd` object containing the result of a mass model fit
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param plot: Plot the gas mass profile. Defaults to False
    :type plot: bool
    :param outfile: If plot=True, file name to output the plotted Mgas profile. If none, the plot is displayed on stdout. Defaults to None
    :type outfile: str
    :param nmore: Number of points defining fine grid, must be equal to the value used for the mass reconstruction. Defaults to 5
    :type nmore: int
    :return: 1-D arrays containing the median gas mass and 16th and 84th percentiles. If plot=True, a matplotlib figure is also returned.
    """

    if Mhyd.samples is None or Mhyd.redshift is None or Mhyd.ccf is None:

        print('Error: no mass reconstruction found')

        return

    nsamp = len(Mhyd.samples)

    rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore=nmore)

    nvalm = len(rin_m)

    if Mhyd.cf_prof is not None:

        cf_prof = np.repeat(Mhyd.cf_prof, nsamp).reshape(nvalm, nsamp)

    else:

        cf_prof = Mhyd.ccf

    alldens = np.sqrt(np.dot(Mhyd.Kdens_m, np.exp(Mhyd.samples.T)) / cf_prof * Mhyd.transf)

    # Matrix containing integration volumes
    volmat = np.repeat(4. / 3. * np.pi * (rout_m ** 3 - rin_m ** 3), nsamp).reshape(nvalm, nsamp)

    # Compute Mgas profile as cumulative sum over the volume

    nhconv = cgsamu * Mhyd.mu_e * cgskpc ** 3 / Msun  # Msun/kpc^3

    ones_mat = np.ones((nvalm, nvalm))

    cs_mat = np.tril(ones_mat)

    mgas = np.dot(cs_mat, alldens * nhconv * volmat)

    mg, mgl, mgh = np.percentile(mgas, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

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

        plt.plot(rout_m, mg, color='blue')

        plt.fill_between(rout_m, mgl, mgh, color='blue', alpha=0.4)

        plt.xlabel('Radius [kpc]', fontsize=40)

        plt.ylabel('$M_{gas} [M_\odot]$', fontsize=40)

        if outfile is not None:
            plt.savefig(outfile)
            plt.close()

        return mg, mgl, mgh, fig

    else:

        return mg, mgl, mgh