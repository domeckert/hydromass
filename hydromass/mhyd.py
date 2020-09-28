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

def Run_Mhyd_PyMC3(Mhyd,model,bkglim=None,nmcmc=1000,fit_bkg=False,back=None,
                   samplefile=None,nrc=None,nbetas=6,min_beta=0.6, nmore=5,
                   p0_prior=None, tune=500, dmonly=False, mstar=None, find_map=True,
                   pnt=False):
    """

    Set up hydrostatic mass model and optimize with PyMC3

    :param Mhyd: (Mhyd) Object including the loaded data and initial setup (mandatory input)
    :param model: (Model) Object including the chosen mass model and its input values (mandatory input)
    :param bkglim: (float) Limit (in arcmin) out to which the SB data will be fitted; if None then the whole range is considered (default = None)
    :param nmcmc: (integer) Number of PyMC3 steps (default = 1000)
    :param fit_bkg: (boolean) Choose whether the counts and the background will be fitted on-the-fly using a Poisson model (fit_bkg=True) or if the surface brightness will be fitted, in which case it is assumed that the background has already been subtracted and Gaussian likelihood will be used (default = False)
    :param back: (float) Input value for the background. If back = None then the mean surface brightness in the region outside "bkglim" is used. Relevant only if fit_bkg = True (default = None).
    :param samplefile: (string) Name of ASCII file to output the final PyMC3 samples
    :param nrc: (integer) Number of core radii values to set up the multiscale model (default = number of data points / 4)
    :param nbetas: (integer) Number of beta values to set up the multiscale model (default = 6)
    :param min_beta: (float) Minimum beta value (default = 0.6)
    :return:

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
    rin_m, rout_m, index_x, index_sz, sum_mat = rads_more(Mhyd, nmore=nmore)

    if dmonly and mstar is not None:

        r_mstar = mstar[:, 0]

        cum_mstar = mstar[:, 1]

        mstar_m = np.interp(rout_m, r_mstar, cum_mstar)

    nptmore = len(rout_m)

    int_mat = cumsum_mat(nptmore)

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

    else:

        Kdens_m = calc_density_operator(rout_m / Mhyd.amin2kpc, pardens, Mhyd.amin2kpc, withbkg=False)

    hydro_model = pm.Model()

    cf = Mhyd.ccf

    if pnt:

        if model.massmod != 'NFW':

            print('Non-thermal pressure correction is currently implemented only for the NFW model, reverting to thermal only')

            pnt = False

        else:

            file_means = get_data_file_path('pnt_mean.dat')

            file_cov = get_data_file_path('pnt_covmat.dat')

            #pnt_mean = np.loadtxt(file_means).astype(np.float32)

            pnt_mean = np.array([[0.15288539, 1.07588625, 0.05624287]])

            #pnt_cov = np.loadtxt(file_cov).astype(np.float32)

            pnt_cov = np.array([[ 0.00400375, -0.0199079, -0.0022997 ], [-0.0199079, 0.54718882, 0.02912699],[-0.0022997, 0.02912699, 0.00343748]])

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

            P0_est = 1e-4 * 5. # 5 keV and R500 density

            err_P0_est = P0_est # 1-dex

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
        mass = Mhyd.mfact * model.func_pm(rout_m, *pmod, delta=model.delta) / Mhyd.mfact0

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
        dpres = - mass / rout_m ** 2 * dens_m * (rout_m - rin_m)

        press_out = press00 - pm.math.dot(int_mat, dpres)  # directly returns press_out

        if pnt:

            c200 = pmod[0]

            r200c = pmod[1]

            alpha_turb = alpha_turb_pm(rout_m, r200c, c200, Mhyd.redshift, pnt_pars)

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

    print('Running MCMC...')

    with hydro_model:

        if find_map:

            start = pm.find_MAP()

            trace = pm.sample(nmcmc, start=start, tune=tune)

        else:

            trace = pm.sample(nmcmc, tune=tune)

    print('Done.')

    tend = time.time()

    print(' Total computing time is: ', (tend - tinit) / 60., ' minutes')

    Mhyd.trace = trace

    # Get chains and save them to file
    sampc = trace.get_values('coefs')

    if fit_bkg:

        sampb = trace.get_values('bkg')

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
    Mhyd.dmonly = dmonly
    Mhyd.mstar = mstar
    Mhyd.pnt = pnt
    if pnt:
        Mhyd.pnt_pars = trace.get_values('Pnt')[:, 0]

    alldens = np.sqrt(np.dot(Kdens, np.exp(samples.T)) / cf * transf)
    pmc = np.median(alldens, axis=1)
    pmcl = np.percentile(alldens, 50. - 68.3 / 2., axis=1)
    pmch = np.percentile(alldens, 50. + 68.3 / 2., axis=1)
    Mhyd.dens = pmc
    Mhyd.dens_lo = pmcl
    Mhyd.dens_hi = pmch

    samppar = np.empty((len(samples), model.npar))
    for i in range(model.npar):

        name = model.parnames[i]
        samppar[:, i] = trace.get_values(name)

    samplogp0 = trace.get_values('logp0')

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



class Mhyd:
    """
    Class Mhyd

    """

    def __init__(self, sbprofile=None, spec_data=None, sz_data=None, directory=None, redshift=None, cosmo=None, f_abund = 'angr'):
        """

        Constructor from class Mhyd

        :param sbprofile: (pyproffit.Profile) Object including the surface brightness data
        :param spec_data: (string) Path to FITS file including results of surface brightness reconstruction
        :param directory: (string) Name of output file directory (default='mhyd')
        :param redshift: (float) Source redshift
        :param cosmo: (astropy.cosmology) Astropy cosmological model
        :param f_abund: (string) Solar abundance table. Available are 'angr', 'aspl', and 'grsa' (default = 'angr')
        """

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


    def emissivity(self, nh, rmf, kt=None, Z=0.3, elow=0.5, ehigh=2.0, arf=None):

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
                                        Z=Z,
                                        elow=elow,
                                        ehigh=ehigh,
                                        arf=arf)


    def run(self, model=None, bkglim=None, nmcmc=1000, fit_bkg=False, back=None,
            samplefile=None, nrc=None, nbetas=6, min_beta=0.6, nmore=5,
            p0_prior=None, tune=500, dmonly=False, mstar=None, find_map=True, pnt=False):

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
                       pnt=pnt)



    def run_forward(self, forward=None, bkglim=None, nmcmc=1000, fit_bkg=False, back=None,
            samplefile=None, nrc=None, nbetas=6, min_beta=0.6, nmore=5, tune=500, find_map=True):

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

    def run_polytropic(self, polytropic=None, bkglim=None, nmcmc=1000, fit_bkg=False, back=None,
            samplefile=None, nrc=None, nbetas=6, min_beta=0.6, nmore=5, tune=500, find_map=True):

        if polytropic is None:

            print('Error no polytropic model provided')

            return


        Run_Polytropic_PyMC3(self,
                          Polytropic=polytropic,
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

        SaveModel(self,
                  model,
                  outfile)

    def SaveGP(self, outfile=None):

        SaveGP(self,
               outfile)

    def SaveForward(self, Forward, outfile=None):

        SaveForward(self,
                    Forward,
                    outfile)