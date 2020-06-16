import numpy as np
import pymc3 as pm
from .deproject import *
from .plots import rads_more
from .constants import *

# GNFW function should work both for numpy.ndarray and pymc3/theano formats
def gnfw_pm(rad, p0, c500, gamma, alfa, beta):

    x = c500 * rad

    t1 = x ** gamma

    t2 = (1. + x ** alfa) ** ( (beta - gamma) / alfa)

    fgnfw = p0 / t1 / t2

    return  fgnfw


def gnfw_np(xout, pars):

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

    Compute model temperature profile from Forward Mhyd reconstruction evaluated at reference X-ray temperature radii

    :param Mhyd: mhyd.Mhyd object including the reconstruction
    :param model: mhyd.Model object defining the mass model
    :return: Median temperature, Lower 1-sigma percentile, Upper 1-sigma percentile
    """

    if Mhyd.temp_x is None:

        print('No spectral data provided')

        return

    nsamp = len(Mhyd.samples)

    rin_m, rout_m, index_x, index_sz = rads_more(Mhyd, nmore=nmore)

    vx = MyDeprojVol(rin_m / Mhyd.amin2kpc, rout_m / Mhyd.amin2kpc)

    vol_x = vx.deproj_vol().T

    npx = len(Mhyd.rref_x)

    dens_m = np.sqrt(np.dot(Mhyd.Kdens_m, np.exp(Mhyd.samples.T)) / Mhyd.ccf * Mhyd.transf)

    p3d = Forward.func_np(rout_m, Mhyd.samppar)

    t3d = p3d / dens_m

    # Mazzotta weights
    ei = dens_m ** 2 * t3d ** (-0.75)

    # Temperature projection
    flux = np.dot(vol_x, ei)

    tproj = np.dot(vol_x, t3d * ei) / flux

    tfit_proj, t3d_or = np.empty((npx, nsamp)), np.empty((npx, nsamp))

    for i in range(nsamp):

        tfit_proj[:, i] = tproj[:, i][index_x]

        t3d_or[:, i] = t3d[:, i][index_x]

    tmed, tlo, thi = np.percentile(tfit_proj, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    t3do, t3dl, t3dh = np.percentile(t3d_or,[50., 50. - 68.3 / 2. , 50. + 68.3 / 2.] , axis=1)

    dict = {
        "R_IN": Mhyd.rin_x,
        "R_OUT": Mhyd.rout_x,
        "R_REF": Mhyd.rref_x,
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

    Compute model pressure profile from Forward Mhyd reconstruction evaluated at the reference SZ radii

    :param Mhyd: mhyd.Mhyd object including the reconstruction
    :param model: mhyd.Model object defining the mass model
    :return: Median pressure, Lower 1-sigma percentile, Upper 1-sigma percentile
    """

    if Mhyd.pres_sz is None:

        print('No SZ data provided')

        return

    nsamp = len(Mhyd.samples)

    rin_m, rout_m, index_x, index_sz = rads_more(Mhyd, nmore=nmore)

    npx = len(Mhyd.rref_sz)

    p3d = Forward.func_np(rout_m, Mhyd.samppar)

    pfit = np.empty((npx, nsamp))

    for i in range(nsamp):

        pfit[:, i] = p3d[:, i][index_sz]

    pmed, plo, phi = np.percentile(pfit, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    return pmed, plo, phi


def mass_forw_from_samples(Mhyd, Forward, plot=False, nmore=5):

    nsamp = len(Mhyd.samples)

    rin_m, rout_m, index_x, index_sz = rads_more(Mhyd, nmore=nmore)

    nvalm = len(rin_m)

    dens_m = np.sqrt(np.dot(Mhyd.Kdens_m, np.exp(Mhyd.samples.T)) / Mhyd.ccf * Mhyd.transf)

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


class Forward:
    """
    Class for definition of forward model to the pressure using a GNFW model
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

    :param Mhyd:
    :param bkglim:
    :param nmcmc:
    :param fit_bkg:
    :param back:
    :param samplefile:
    :param nrc:
    :param nbetas:
    :param min_beta:
    :param tune:
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
    rin_m, rout_m, index_x, index_sz = rads_more(Mhyd, nmore=nmore)

    nptmore = len(rout_m)

    vx = MyDeprojVol(rin_m / Mhyd.amin2kpc, rout_m / Mhyd.amin2kpc)

    vol = vx.deproj_vol().T

    if fit_bkg:

        Kdens_m = calc_density_operator(rout_m / Mhyd.amin2kpc, pardens, Mhyd.amin2kpc)

    else:

        Kdens_m = calc_density_operator(rout_m / Mhyd.amin2kpc, pardens, Mhyd.amin2kpc, withbkg=False)

    hydro_model = pm.Model()

    cf = Mhyd.ccf

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

        for i in range(Forward.npar):

            name = Forward.parnames[i]

            if not Forward.fix[i]:

                lim = Forward.limits[i]

                if name == 'p0':

                    tpar = pm.TruncatedNormal(name, mu=np.log(Forward.start[i]), sd=Forward.sd[i] / Forward.start[i],
                                                lower=np.log(lim[0]), upper=np.log(lim[1])) #log-normal prior on normalization

                    modpar = np.exp(tpar)

                else:

                    modpar = pm.TruncatedNormal(name, mu=Forward.start[i], sd=Forward.sd[i],
                                                lower=lim[0], upper=lim[1]) #Gaussian prior on other parameters
            else:

                dummy = pm.Normal('dummy'+name, mu=0., sd=1.)

                dummy_param = 0 * dummy + Forward.start[i]

                modpar = pm.Deterministic(name, dummy_param)

            allpmod.append(modpar)

        for RV in hydro_model.basic_RVs:
            print(RV.name, RV.logp(hydro_model.test_point))

        pmod = pm.math.stack(allpmod, axis=0)

        dens_m = pm.math.sqrt(pm.math.dot(Kdens_m, al) / cf * transf)  # electron density in cm-3

        p3d = Forward.func_pm(rout_m, *pmod)

        # Model temperature
        t3d = p3d / dens_m

        # Mazzotta weights
        ei = dens_m ** 2 * t3d ** (-0.75)

        # Temperature projection
        flux = pm.math.dot(vol, ei)

        tproj = pm.math.dot(vol, t3d * ei) / flux

        # Density Likelihood
        if fit_bkg:

            count_obs = pm.Poisson('counts', mu=pred, observed=counts)  # counts likelihood

        else:

            sb_obs = pm.Normal('sb', mu=pred, observed=sb, sd=esb)  # Sx likelihood

        # Temperature model and likelihood
        if Mhyd.temp_x is not None:
            tfit_proj = tproj[index_x]

            T_obs = pm.Normal('kt', mu=tfit_proj, observed=Mhyd.temp_x, sd=Mhyd.errt_x)  # temperature likelihood

        # SZ pressure model and likelihood
        if Mhyd.pres_sz is not None:
            pfit = p3d[index_sz]

            P_obs = pm.MvNormal('P', mu=pfit, observed=Mhyd.pres_sz, cov=Mhyd.covmat_sz)  # SZ pressure likelihood

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

        samppar = np.empty((len(samples), Forward.npar))
        for i in range(Forward.npar):

            name = Forward.parnames[i]

            if name == 'p0':

                samppar[:, i] = np.exp(trace.get_values(name))

            else:
                samppar[:, i] = trace.get_values(name)
        Mhyd.samppar = samppar

        Mhyd.K = K
        Mhyd.Kdens = Kdens
        Mhyd.Ksb = Ksb
        Mhyd.transf = transf
        Mhyd.Kdens_m = Kdens_m

        if Mhyd.temp_x is not None:
            kt_mod = kt_forw_from_samples(Mhyd, Forward, nmore=nmore)
            Mhyd.ktmod = kt_mod['TSPEC']
            Mhyd.ktmod_lo = kt_mod['TSPEC_LO']
            Mhyd.ktmod_hi = kt_mod['TSPEC_HI']
            Mhyd.kt3d = kt_mod['T3D']
            Mhyd.kt3d_lo = kt_mod['T3D_LO']
            Mhyd.kt3d_hi = kt_mod['T3D_HI']

        if Mhyd.pres_sz is not None:
            pmed, plo, phi = P_forw_from_samples(Mhyd, Forward, nmore=nmore)
            Mhyd.pmod = pmed
            Mhyd.pmod_lo = plo
            Mhyd.pmod_hi = phi
