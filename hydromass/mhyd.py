from .deproject import *
from .emissivity import *
from .functions import *
from .plots import *
from .constants import *
from .forward import *
from .polytropic import *
from .pnt import *
from .nonparametric import *
from astropy.io import fits
import os
import pymc3 as pm
from .save import *
import arviz as az

def Run_Mhyd_PyMC3(Mhyd,model,bkglim=None,nmcmc=1000,fit_bkg=False,back=None,
                   samplefile=None,nrc=None,nbetas=6,min_beta=0.6, nmore=5,
                   p0_prior=None, tune=500, dmonly=False, mstar=None, find_map=True,
                   pnt=False, rmin=None, rmax=None):
    """

    Set up hydrostatic mass model and optimize with PyMC3. The routine takes a parametric mass model as input and integrates the hydrostatic equilibrium equation to predict the 3D pressure profile:

    .. math::

        P_{3D}(r) = P_0 + \\int_{r}^{r_0} \\rho_{gas}(r) \\frac{G M_{mod}(<r)}{r^2} dr

    with :math:`r_0` the outer radial boundary of the input data and :math:`P_0` the pressure at :math:`r_0`.

    The gas density profile is fitted to the surface brightness profile and described as a linear combination of King functions. The mass model should be defined using the :class:`hydromass.functions.Model` class, which implements a number of popular mass models. The 3D mass profile is then projected along the line of sight an weighted by spectroscopic-like weights to predict the spectroscopic temperature profile.

    The parameters of the mass model and of the gas density profile are fitted jointly to the data. Priors on the input parameters can be set by the user in the definition of the mass model.

    :param Mhyd: A :class:`hydromass.mhyd.Mhyd` object including the loaded data and initial setup (mandatory input)
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param model:  A :class:`hydromass.functions.Model` object including the chosen mass model and its input values (mandatory input)
    :type model: class:`hydromass.functions.Model`
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
    :param p0_prior: Set of two values defining the mean and standard deviation of the Gaussian prior on p0. If None, the code attempts to determine the value of P0 using the :func:`hydromass.plots.estimate_P0` function, which fits a rough gNFW function to estimate the shape of the pressure profile and uses the fitted function to approximate the value of P0.
    :type p0_prior: numpy.ndarray
    :param tune: Number of NUTS tuning steps. Defaults to 500
    :type tune: int
    :param dmonly: Specify whether the mass model is fitted to the total mass (dmonly=False) or to the dark matter only after subtracting the gas mass and the stellar mass if provided (dmonly=True). Defaults to False.
    :type dmonly: bool
    :param mstar: If dmonly=True, provide an array containing the cumulative stellar mass profile, which will be subtracted when adjusting the mass model to the dark matter only.
    :type mstar: numpy.ndarray
    :param find_map: Specify whether a maximum likelihood fit will be performed first to initiate the sampler. Defaults to True
    :type find_map: bool
    :param pnt: Attempt to model the non-thermal pressure profile. If pnt=True, the non-thermal pressure profile of Angelinelli et al. 2020 and the corresponding parameters are used to marginalize over the impact of non-thermal pressure. Defaults to False.
    :type pnt: bool
    :param rmin: Minimum limiting radius (in arcmin) of the active region for the surface brightness. If rmin=None, no minimum radius is applied. Defaults to None.
    :type rmin: float
    :param rmax: Maximum limiting radius (in arcmin) of the active region for the surface brightness. If rmax=None, no maximum radius is applied. Defaults to None.
    :type rmax: float

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

    if rmin is not None:
        valid = np.where(rad>=rmin)
        sb = sb[valid]
        esb = esb[valid]
        rad = rad[valid]
        erad = erad[valid]
        counts = counts[valid]
        area = area[valid]
        exposure = exposure[valid]
        bkgcounts = bkgcounts[valid]

    if rmax is not None:
        valid = np.where(rad <= rmax)
        sb = sb[valid]
        esb = esb[valid]
        rad = rad[valid]
        erad = erad[valid]
        counts = counts[valid]
        area = area[valid]
        exposure = exposure[valid]
        bkgcounts = bkgcounts[valid]


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

        K = np.dot(prof.psfmat, Ksb)
        # K = calc_sb_operator_psf(rad, sourcereg, pars, area, exposure, psfmat) # transformation to surface brightness

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

    rref_m = (rin_m + rout_m)/2.

    if dmonly and mstar is not None:

        r_mstar = mstar[:, 0]

        cum_mstar = mstar[:, 1]

        mstar_m = np.interp(rout_m, r_mstar, cum_mstar)

    nptmore = len(rout_m)

    int_mat = cumsum_mat(nptmore)

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

        Kdens_m = calc_density_operator(rref_m / Mhyd.amin2kpc, pardens, Mhyd.amin2kpc)

    else:

        Kdens_m = calc_density_operator(rref_m / Mhyd.amin2kpc, pardens, Mhyd.amin2kpc, withbkg=False)

    hydro_model = pm.Model()

    if pnt:

        if model.massmod != 'NFW':

            print('Non-thermal pressure correction is currently implemented only for the NFW model, reverting to thermal only')

            pnt = False

        else:

            file_means = get_data_file_path('pnt_mean.dat')

            file_cov = get_data_file_path('pnt_covmat.dat')

            pnt_mean = np.loadtxt(file_means).astype(np.float32)

            #pnt_mean = np.array([[0.15288539, 1.07588625, 0.05624287]])

            pnt_cov = np.loadtxt(file_cov).astype(np.float32)

            #pnt_cov = np.array([[ 0.00400375, -0.0199079, -0.0022997 ], [-0.0199079, 0.54718882, 0.02912699],[-0.0022997, 0.02912699, 0.00343748]])

    with hydro_model:
        # Priors for unknown model parameters
        coefs = pm.Normal('coefs', mu=testval, sd=20, shape=npt)

        if fit_bkg:

            bkgd = pm.Normal('bkg', mu=testbkg, sd=0.05, shape=1) # in case fit_bkg = False this is not fitted

            ctot = pm.math.concatenate((coefs, bkgd), axis=0)

            al = pm.math.exp(ctot)

            pred = pm.math.dot(K, al) + bkgcounts  # Predicted number of counts per annulus

        else:

            al = pm.math.exp(coefs)

            pred = pm.math.dot(K, al)

        # Model parameters
        allpmod = []

        for i in range(model.npar):

            name = model.parnames[i]

            if not model.fix[i]:

                lim = model.limits[i]

                modpar = pm.TruncatedNormal(name, mu=model.start[i], sd=model.sd[i], lower=lim[0], upper=lim[1]) #

            else:

                modpar = pm.ConstantDist(name, model.start[i])

            allpmod.append(modpar)

        pmod = pm.math.stack(allpmod, axis=0)

        # Integration constant as a fitting (nuisance) parameter
        if p0_prior is not None:

            P0_est = p0_prior[0]

            err_P0_est = p0_prior[1]

        else:

            P0_est = estimate_P0(Mhyd=Mhyd)

            print('Estimated value of P0: %g' % (P0_est))

            err_P0_est = P0_est # 1 in ln

        logp0 = pm.TruncatedNormal('logp0', mu=np.log(P0_est), sd=err_P0_est / P0_est,
                                   lower=np.log(P0_est) - err_P0_est / P0_est,
                                   upper=np.log(P0_est) + err_P0_est / P0_est)

        if pnt:

            pnt_pars = pm.MvNormal('Pnt', mu=pnt_mean, cov=pnt_cov, shape=(1,3))

        for RV in hydro_model.basic_RVs:
            print(RV.name, RV.logp(hydro_model.test_point))

        press00 = np.exp(logp0)

        dens_m = pm.math.sqrt(pm.math.dot(Kdens_m, al) / cf * transf)  # electron density in cm-3

        # Evaluate mass model
        mass = Mhyd.mfact * model.func_pm(rref_m, *pmod, delta=model.delta) / Mhyd.mfact0

        if dmonly:

            nhconv = cgsamu * Mhyd.mu_e * cgskpc ** 3 / Msun  # Msun/kpc^3

            mgas = mgas_pm(rin_m, rout_m, dens_m) * nhconv / Mhyd.mfact0

            # Add stellar mass if provided
            if mstar is not None:

                mbar = mgas + mstar_m / Mhyd.mfact0 / 1e13

            else:

                mbar = mgas

            mass = mass + mbar

        # Pressure gradient
        dpres = - mass / rref_m ** 2 * dens_m * (rout_m - rin_m)

        press_out = press00 - pm.math.dot(int_mat, dpres)  # directly returns press_out

        # Non-thermal pressure correction, if any
        if pnt:

            c200 = pmod[0]

            r200c = pmod[1]

            alpha_turb = alpha_turb_pm(rref_m, r200c, c200, Mhyd.redshift, pnt_pars)

            pth = press_out * (1. - alpha_turb)

        else:

            pth = press_out

        # Density Likelihood
        if fit_bkg:

            count_obs = pm.Poisson('counts', mu=pred, observed=counts) #counts likelihood

        else:

            sb_obs = pm.Normal('sb', mu=pred, observed=sb, sd=esb) #Sx likelihood

        # Temperature model and likelihood
        if Mhyd.spec_data is not None:

            # Model temperature
            t3d = pth / dens_m

            # Mazzotta weights
            ei = dens_m ** 2 * t3d ** (-0.75)

            # Temperature projection
            flux = pm.math.dot(proj_mat, ei)

            tproj = pm.math.dot(proj_mat, t3d * ei) / flux

            T_obs = pm.Normal('kt', mu=tproj, observed=Mhyd.spec_data.temp_x, sd=Mhyd.spec_data.errt_x)  # temperature likelihood

        # SZ pressure model and likelihood
        if Mhyd.sz_data is not None:

            pfit = pth[index_sz]

            P_obs = pm.MvNormal('P', mu=pfit, observed=Mhyd.sz_data.pres_sz, cov=Mhyd.sz_data.covmat_sz)  # SZ pressure likelihood


    tinit = time.time()

    print('Running HMC...')

    with hydro_model:

        if find_map:

            start = pm.find_MAP()

            trace = pm.sample(nmcmc, init='ADVI', start=start, tune=tune, return_inferencedata=True,
                              target_accept=0.9)

        else:

            trace = pm.sample(nmcmc, init='ADVI', tune=tune, return_inferencedata=True, target_accept=0.9)


        Mhyd.ppc_sb = pm.sample_posterior_predictive(trace, var_names=['sb'])

        if Mhyd.spec_data is not None:

            Mhyd.ppc_kt = pm.sample_posterior_predictive(trace, var_names=['kt'])

        if Mhyd.sz_data is not None:

            Mhyd.ppc_sz = pm.sample_posterior_predictive(trace, var_names=['P'])

    print('Done.')

    tend = time.time()

    print(' Total computing time is: ', (tend - tinit) / 60., ' minutes')

    Mhyd.trace = trace

    Mhyd.hydro_model = hydro_model

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
        np.savetxt(samplefile+'.par',np.array([pars.shape[0]/nbetas,nbetas,min_beta,nmcmc]),header='pymc3')

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
    Mhyd.dmonly = dmonly
    Mhyd.mstar = mstar
    Mhyd.pnt = pnt
    if pnt:
        Mhyd.pnt_pars = np.array(trace.posterior['Pnt']).reshape(sc_coefs[0] * sc_coefs[1], 3)

    alldens = np.sqrt(np.dot(Kdens, np.exp(samples.T)) * transf)
    pmc = np.median(alldens, axis=1) / np.sqrt(Mhyd.ccf)
    pmcl = np.percentile(alldens, 50. - 68.3 / 2., axis=1) / np.sqrt(Mhyd.ccf)
    pmch = np.percentile(alldens, 50. + 68.3 / 2., axis=1) / np.sqrt(Mhyd.ccf)
    Mhyd.dens = pmc
    Mhyd.dens_lo = pmcl
    Mhyd.dens_hi = pmch

    samppar = np.empty((len(samples), model.npar))
    for i in range(model.npar):

        name = model.parnames[i]
        samppar[:, i] = np.array(trace.posterior[name]).flatten()

    samplogp0 = np.array(trace.posterior['logp0']).flatten()

    Mhyd.samppar = samppar
    Mhyd.samplogp0 = samplogp0
    Mhyd.K = K
    Mhyd.Kdens = Kdens
    Mhyd.Ksb = Ksb
    Mhyd.transf = transf
    Mhyd.Kdens_m = Kdens_m

    if Mhyd.spec_data is not None:
        kt_mod = kt_from_samples(Mhyd, model, nmore=nmore)
        Mhyd.ktmod = kt_mod['TSPEC']
        Mhyd.ktmod_lo = kt_mod['TSPEC_LO']
        Mhyd.ktmod_hi = kt_mod['TSPEC_HI']
        Mhyd.kt3d = kt_mod['T3D']
        Mhyd.kt3d_lo = kt_mod['T3D_LO']
        Mhyd.kt3d_hi = kt_mod['T3D_HI']

    if Mhyd.sz_data is not None:
        pmed, plo, phi = P_from_samples(Mhyd, model, nmore=nmore)
        Mhyd.pmod = pmed
        Mhyd.pmod_lo = plo
        Mhyd.pmod_hi = phi

    totlike = 0.
    nptot = model.npar + 1
    thermolike = 0.
    npthermo = model.npar + 1
    Mhyd.trace.log_likelihood['tot'] = 0.

    if fit_bkg:
        totlike = totlike + np.sum(np.asarray(Mhyd.trace['log_likelihood']['counts']), axis=2).flatten()
        nptot = nptot + npt + 1
        #Mhyd.trace.log_likelihood['tot'] = Mhyd.trace.log_likelihood['counts']

    else:
        totlike = totlike + np.sum(np.asarray(Mhyd.trace['log_likelihood']['sb']), axis=2).flatten()
        nptot = nptot + npt
        #Mhyd.trace.log_likelihood['tot'] = Mhyd.trace.log_likelihood['sb']

    if Mhyd.spec_data is not None:
        totlike = totlike + np.sum(np.asarray(Mhyd.trace['log_likelihood']['kt']), axis=2).flatten()
        thermolike = thermolike + np.sum(np.asarray(Mhyd.trace['log_likelihood']['kt']), axis=2).flatten()
        Mhyd.trace.log_likelihood['tot'] = Mhyd.trace.log_likelihood['tot'] + Mhyd.trace.log_likelihood['kt']


    if Mhyd.sz_data is not None:
        totlike = totlike + np.sum(np.asarray(Mhyd.trace['log_likelihood']['P']), axis=2).flatten()
        thermolike = thermolike + np.sum(np.asarray(Mhyd.trace['log_likelihood']['P']), axis=2).flatten()
        Mhyd.trace.log_likelihood['tot'] = Mhyd.trace.log_likelihood['tot'] + Mhyd.trace.log_likelihood['P']

    if pnt:
        nptot = nptot + 3
        npthermo = npthermo + 3

    Mhyd.totlike = totlike
    Mhyd.nptot = nptot
    Mhyd.thermolike = thermolike
    Mhyd.npthermo = npthermo
    Mhyd.waic = az.waic(Mhyd.trace, var_name='tot')
    Mhyd.loo = az.loo(Mhyd.trace, var_name='tot')


class Mhyd:
    """

    The Mhyd class is the core class of hydromass. It allows the user to pass one or more datasets to be fitted, chose the mass reconstruction method, set options like change cosmology, Solar abundance table, NUTS options, model choice, and more.

    :param sbprofile: A pyproffit Profile object (https://pyproffit.readthedocs.io/en/latest/pyproffit.html#module-pyproffit.profextract) including the surface brightness data
    :type sbprofile: class:`pyproffit.profextract.Profile`
    :param spec_data: A :class:`hydromass.tpdata.SpecData` object including a spectroscopic X-ray temperature profile and its associated uncertainties
    :type spec_data: class:`hydromass.tpdata.SpecData`
    :param sz_data: A :class:`hydromass.tpdata.SZData` object containing an SZ pressure profile and its covariance matrix
    :type sz_data: class:`hydromass.tpdata.SZData`
    :param directory: Name of output file directory. Defaults to 'mhyd'
    :type directory: str
    :param redshift: Source redshift
    :type redshift: float
    :param cosmo: Astropy cosmological model
    :type cosmo: class:`astropy.cosmology`
    :param f_abund: Solar abundance table. Available are 'angr', 'aspl', and 'grsa'. Defaults to 'angr'
    :type f_abund: str
    """

    def __init__(self, sbprofile=None, spec_data=None, sz_data=None, directory=None, redshift=None, cosmo=None, f_abund = 'angr'):

        if f_abund == 'angr':
            nhc = 1 / 0.8337
            mup = 0.6125
            mu_e = 1.1738
        elif f_abund == 'aspl':
            nhc = 1 / 0.8527
            mup = 0.5994
            mu_e = 1.1548
        elif f_abund == 'grsa':
            nhc = 1 / 0.8520
            mup = 0.6000
            mu_e = 1.1555
        else:  # aspl default
            nhc = 1 / 0.8527
            mup = 0.5994
            mu_e = 1.1548
        self.nhc=nhc
        self.mup=mup
        self.mu_e=mu_e

        if directory is None:

            print('No output directory name provided, will output to subdirectory "mhyd" ')

            directory = 'mhyd'

        if not os.path.exists(directory):

            os.makedirs(directory)

        self.dir = directory

        if sbprofile is None:

            print('Error: no surface brightness profile provided, please provide one with the "sbprofile=" option')

            return

        self.sbprof = sbprofile

        if redshift is None:

            print('Error: no redshift provided, please provide one with the "redshift=" option')
            return

        self.redshift = redshift

        if cosmo is None:

            print('No cosmology provided, will default to Planck15')

            from astropy.cosmology import Planck15 as cosmo

        self.cosmo = cosmo

        dlum = cosmo.luminosity_distance(redshift)

        self.dlum = np.asarray(dlum, dtype=float)

        print('Luminosity distance to the source: %g Mpc' % (self.dlum))

        amin2kpc = cosmo.kpc_proper_per_arcmin(redshift).value

        self.amin2kpc = amin2kpc

        print('At the redshift of the source 1 arcmin is %g kpc' % (self.amin2kpc))

        if spec_data is None and sz_data is None:

            print('Error: no spectral data file or SZ data file provided, please provide at least one with the "spec_data=" or "sz_data=" options')

            return

        if spec_data is None and sz_data is None:

            print('Error: no spectral data file or SZ data file provided, please provide at least one with the "spec_data=" or "sz_data=" options')

            return

        if spec_data is not None:

            self.spec_data = spec_data

        else:

            self.spec_data = None

        if sz_data is not None:

            self.sz_data = sz_data

        else:

            self.sz_data = None

        rho_cz = cosmo.critical_density(self.redshift).value * cgsMpc ** 3 / Msun # critical density in Msun per Mpc^3

        self.mfact = 4. * np.pi * rho_cz * 1e-22

        self.mfact0 = kev2erg * cgskpc / (cgsG * cgsamu * self.mup) / Msun / 1e13

        self.mgas_fact = cgsamu * self.mu_e / Msun


    def emissivity(self, nh, rmf, kt=None, abund='angr', Z=0.3, elow=0.5, ehigh=2.0, arf=None):
        '''
        Compute the conversion between count rate and emissivity using XSPEC by run the :func:`hydromass.emissivity.calc_emissivity` function. Requires XSPEC to be available in PATH.

        :param nh: Source NH in units of 1e22 cm**(-2)
        :type nh: float
        :param kt: Source temperature in keV. If None, the code will search for a loaded spectroscopic temperature profile and use the weighted mean temperature. Defaults to None
        :type kt: float
        :param rmf: Path to response file (RMF/RSP)
        :type rmf: str
        :param abund: Solar abundance table in XSPEC format. Defaults to "angr"
        :type abund: str
        :param Z: Metallicity with respect to solar. Defaults to 0.3
        :type Z: float
        :param elow: Low-energy bound of the input image in keV. Defaults to 0.5
        :type elow: float
        :param ehigh: High-energy bound of the input image in keV. Defaults to 2.0
        :type ehigh: float
        :param arf: Path to on-axis ARF (optional, in case response file is RMF)
        :type arf: str
        '''

        if kt is None:

            if self.spec_data.temp_x is not None:

                kt = np.average(self.spec_data.temp_x, weights=1. / self.spec_data.errt_x ** 2)

            else:

                print('Error: no temperature provided, cannot proceed')

                return


        print('Mean cluster temperature:',kt,' keV')

        self.ccf = calc_emissivity(cosmo=self.cosmo,
                                        z=self.redshift,
                                        nh=nh,
                                        kt=kt,
                                        rmf=rmf,
                                        abund=abund,
                                        Z=Z,
                                        elow=elow,
                                        ehigh=ehigh,
                                        arf=arf)


    def run(self, model=None, bkglim=None, nmcmc=1000, fit_bkg=False, back=None,
            samplefile=None, nrc=None, nbetas=6, min_beta=0.6, nmore=5,
            p0_prior=None, tune=500, dmonly=False, mstar=None, find_map=True, pnt=False, rmin=None, rmax=None):
        '''
        Optimize the mass model using the :func:`hydromass.mhyd.Run_Mhyd_PyMC3` function.

        :param model:  A :class:`hydromass.functions.Model` object including the chosen mass model and its input values (mandatory input)
        :type model: class:`hydromass.functions.Model`
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
        :param p0_prior: Set of two values defining the mean and standard deviation of the Gaussian prior on p0. If None, the code attempts to determine the value of P0 using the :func:`hydromass.plots.estimate_P0` function, which fits a rough gNFW function to estimate the shape of the pressure profile and uses the fitted function to approximate the value of P0.
        :type p0_prior: numpy.ndarray
        :param tune: Number of NUTS tuning steps. Defaults to 500
        :type tune: int
        :param dmonly: Specify whether the mass model is fitted to the total mass (dmonly=False) or to the dark matter only after subtracting the gas mass and the stellar mass if provided (dmonly=True). Defaults to False.
        :type dmonly: bool
        :param mstar: If dmonly=True, provide an array containing the cumulative stellar mass profile, which will be subtracted when adjusting the mass model to the dark matter only.
        :type mstar: numpy.ndarray
        :param find_map: Specify whether a maximum likelihood fit will be performed first to initiate the sampler. Defaults to True
        :type find_map: bool
        :param pnt: Attempt to model the non-thermal pressure profile. If pnt=True, the non-thermal pressure profile of Angelinelli et al. 2020 and the corresponding parameters are used to marginalize over the impact of non-thermal pressure. Defaults to False.
        :type pnt: bool
        :param rmin: Minimum limiting radius (in arcmin) of the active region for the surface brightness. If rmin=None, no minimum radius is applied. Defaults to None.
        :type rmin: float
        :param rmax: Maximum limiting radius (in arcmin) of the active region for the surface brightness. If rmax=None, no maximum radius is applied. Defaults to None.
        :type rmax: float
        '''
        if model is None:

            print('Error: No mass model provided')

            return

        Run_Mhyd_PyMC3(self,
                       model=model,
                       bkglim=bkglim,
                       nmcmc=nmcmc,
                       fit_bkg=fit_bkg,
                       back=back,
                       samplefile=samplefile,
                       nrc=nrc,
                       nbetas=nbetas,
                       min_beta=min_beta,
                       nmore=nmore,
                       p0_prior=p0_prior,
                       tune=tune,
                       dmonly=dmonly,
                       mstar=mstar,
                       find_map=find_map,
                       pnt=pnt,
                       rmin=rmin,
                       rmax=rmax)


    def run_forward(self, forward=None, bkglim=None, nmcmc=1000, fit_bkg=False, back=None,
            samplefile=None, nrc=None, nbetas=6, min_beta=0.6, nmore=5, tune=500, find_map=True):

        '''
        Optimize a parametric forward fit to the gas pressure profile using the :func:`hydromass.forward.Run_Forward_PyMC3` function

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
        '''
        if forward is None:

            print('Error no forward model provided')

            return


        Run_Forward_PyMC3(self,
                          Forward=forward,
                          bkglim=bkglim,
                          nmcmc=nmcmc,
                          fit_bkg=fit_bkg,
                          back=back,
                          samplefile=samplefile,
                          nrc=nrc,
                          nbetas=nbetas,
                          min_beta=min_beta,
                          nmore=nmore,
                          tune=tune,
                          find_map=find_map)

    def run_polytropic(self, Polytropic=None, bkglim=None, nmcmc=1000, fit_bkg=False, back=None,
            samplefile=None, nrc=None, nbetas=6, min_beta=0.6, nmore=5, tune=500, find_map=True):
        '''
        Run a polytropic reconstruction with an effective polytropic index model set by the :class:`hydromass.polytropic.Polytropic` class. See :func:`hydromass.polytropic.Run_Polytropic_PyMC3`

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
        '''
        if Polytropic is None:

            print('Error no polytropic model provided')

            return


        Run_Polytropic_PyMC3(self,
                          Polytropic=Polytropic,
                          bkglim=bkglim,
                          nmcmc=nmcmc,
                          fit_bkg=fit_bkg,
                          back=back,
                          samplefile=samplefile,
                          nrc=nrc,
                          nbetas=nbetas,
                          min_beta=min_beta,
                          nmore=nmore,
                          tune=tune,
                          find_map=find_map)


    def run_GP(self, bkglim=None, nmcmc=1000, fit_bkg=False, back=None,
            samplefile=None, nrc=None, nbetas=6, min_beta=0.6, nmore=5, tune=500, find_map=True,
            bin_fact=1.0, smin=None, smax=None, ngauss=100):

        '''
        Run a non-parametric log-normal mixture reconstruction. See :func:`hydromass.nonparametric.Run_NonParametric_PyMC3`

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
        '''

        Run_NonParametric_PyMC3(self,
                                bkglim=bkglim,
                                nmcmc=nmcmc,
                                fit_bkg=fit_bkg,
                                back=back,
                                samplefile=samplefile,
                                nrc=nrc,
                                nbetas=nbetas,
                                min_beta=min_beta,
                                nmore=nmore,
                                tune=tune,
                                find_map=find_map,
                                bin_fact=bin_fact,
                                smin=smin,
                                smax=smax,
                                ngauss=ngauss)

    def SaveModel(self, model, outfile=None):
        '''
        Save the output of a mass model fit into a FITS file through the :func:`hydromass.save.SaveModel` function

        :param model: :class:`hydromass.functions.Model` defining the chosen mass model
        :type model: :class:`hydromass.functions.Model`
        :param outfile: Name of output FITS file. If None, the file is outputted to a file called "output_model.fits" under the default output directory specified in the current object. Defaults to None
        :type outfile: str
        '''

        SaveModel(self,
                  model,
                  outfile)

    def SaveGP(self, outfile=None):
        '''
        Save the output of a non-parametric reconstruction into a FITS file through the :func:`hydromass.save.SaveGP` function

        :param outfile: Name of output FITS file. If None, the file is outputted to a file called "output_GP.fits" under the default output directory specified in the current object. Defaults to None
        :type outfile: str
        '''

        SaveGP(self,
               outfile)

    def SaveForward(self, Forward, outfile=None):
        '''
        Save the output of a parametric forward reconstruction into a FITS file through the :func:`hydromass.save.SaveForward` function

        :param Forward: A :class:`hydromass.forward.Forward` object containing the definition of the forward model
        :type Forward: class:`hydromass.forward.Forward`
        :param outfile: Name of output FITS file. If None, the file is outputted to a file called "output_forward.fits" under the default output directory specified in the current object. Defaults to None
        :type outfile: str
        '''

        SaveForward(self,
                    Forward,
                    outfile)
