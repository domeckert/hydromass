from .plots import *
from .functions import *
from .nonparametric import *
from .forward import *

def SaveModel(Mhyd, model, outfile=None):
    if outfile is None:
        outfile = Mhyd.dir + '/output_model.fits'

    if Mhyd.samples is None:
        print('Nothing to save, exiting')

        return

    if model is None:
        print('No model provided, exiting')
        return

    if model.massmod is None:
        print('Incorrect model provided, exiting')
        return

    hdus = fits.HDUList(hdus=[fits.PrimaryHDU()])

    denshdu = fits.ImageHDU(Mhyd.samples, name='DENSITY')

    headsamp = denshdu.header

    headsamp['BKGLIM'] = Mhyd.bkglim
    headsamp['NRC'] = Mhyd.nrc
    headsamp['NBETAS'] = Mhyd.nbetas
    headsamp['MINBETA'] = Mhyd.min_beta
    headsamp['CCF'] = Mhyd.ccf
    headsamp['TRANSF'] = Mhyd.transf
    headsamp['FITBKG'] = Mhyd.fit_bkg
    headsamp['NMORE'] = Mhyd.nmore
    headsamp['DMONLY'] = Mhyd.dmonly
    headsamp['PNT'] = Mhyd.pnt

    hdus.append(denshdu)

    cols = []
    for i in range(model.npar):
        col = fits.Column(name=model.parnames[i], format='E', array=Mhyd.samppar[:, i])

        cols.append(col)

    col = fits.Column(name='logP0', format='E', array=Mhyd.samplogp0)

    cols.append(col)

    coldefs = fits.ColDefs(cols)

    modhdu = fits.BinTableHDU.from_columns(coldefs)

    modhdu.name = 'MASS MODEL'

    modhead = modhdu.header
    modhead['MASSMOD'] = model.massmod

    modhead['DELTA'] = model.delta

    for i in range(model.npar):
        parname = model.parnames[i]

        modhead['ST' + parname] = model.start[i]

        modhead['SD' + parname] = model.sd[i]

        modhead['LOW' + parname] = model.limits[i][0]

        modhead['HI' + parname] = model.limits[i][1]

    hdus.append(modhdu)

    hdus.writeto(outfile, overwrite=True)


def ReloadModel(Mhyd, infile, mstar=None):
    fin = fits.open(infile)

    Mhyd.samples = fin[1].data

    headden = fin[1].header

    Mhyd.bkglim = headden['BKGLIM']

    Mhyd.nrc = headden['NRC']

    Mhyd.nbetas = headden['NBETAS']

    Mhyd.min_beta = headden['MINBETA']

    Mhyd.ccf = headden['CCF']

    Mhyd.transf = headden['TRANSF']

    Mhyd.fit_bkg = headden['FITBKG']

    Mhyd.nmore = headden['NMORE']

    Mhyd.dmonly = headden['DMONLY']

    Mhyd.mstar = mstar

    Mhyd.pnt = headden['PNT']

    modhead = fin[2].header

    massmod = modhead['MASSMOD']

    delta = modhead['DELTA']

    mod = Model(massmod, delta=delta)

    for i in range(mod.npar):
        parname = mod.parnames[i]

        mod.start[i] = modhead['ST' + parname]

        mod.sd[i] = modhead['SD' + parname]

        mod.limits[i][0] = modhead['LOW' + parname]

        mod.limits[i][1] = modhead['HI' + parname]

    nsamp = int(fin[2].header['NAXIS2'])

    Mhyd.samppar = np.empty((nsamp, mod.npar))

    dpar = fin[2].data

    for i in range(mod.npar):
        name = mod.parnames[i]

        Mhyd.samppar[:, i] = dpar[name]

    Mhyd.samplogp0 = dpar['logP0']

    # Now recreate operators

    prof = Mhyd.sbprof

    rad = prof.bins

    area = prof.area

    exposure = prof.effexp

    sourcereg = np.where(rad < Mhyd.bkglim)

    pars = list_params(rad, sourcereg, Mhyd.nrc, Mhyd.nbetas, Mhyd.min_beta)

    npt = len(pars)

    if prof.psfmat is not None:
        psfmat = np.transpose(prof.psfmat)
    else:
        psfmat = np.eye(prof.nbin)

    Mhyd.pardens = list_params_density(rad, sourcereg, Mhyd.amin2kpc, Mhyd.nrc, Mhyd.nbetas, Mhyd.min_beta)

    # Compute linear combination kernel
    if Mhyd.fit_bkg:

        Mhyd.K = calc_linear_operator(rad, sourcereg, pars, area, exposure,
                                                psfmat)  # transformation to counts
        Mhyd.Kdens = calc_density_operator(rad, Mhyd.pardens, Mhyd.amin2kpc)

    else:

        Mhyd.K = calc_sb_operator_psf(rad, sourcereg, pars, area, exposure,
                                                psfmat)  # transformation to surface brightness
        Mhyd.Kdens = calc_density_operator(rad, Mhyd.pardens, Mhyd.amin2kpc, withbkg=False)

    # Define the fine grid onto which the mass model will be computed
    rin_m, rout_m, index_x, index_sz, sum_mat = rads_more(Mhyd, nmore=Mhyd.nmore)

    if Mhyd.fit_bkg:

        Mhyd.Kdens_m = calc_density_operator(rout_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc)

    else:

        Mhyd.Kdens_m = calc_density_operator(rout_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc,
                                                       withbkg=False)

    if Mhyd.fit_bkg:

        Ksb = calc_sb_operator(rad, sourcereg, pars)

        allsb = np.dot(Ksb, np.exp(Mhyd.samples.T))

        bfit = np.median(np.exp(Mhyd.samples[:, npt]))

        Mhyd.bkg = bfit

    else:

        Ksb = calc_sb_operator(rad, sourcereg, pars, withbkg=False)

        allsb = np.dot(Ksb, np.exp(Mhyd.samples.T))

    pmc = np.median(allsb, axis=1)
    pmcl = np.percentile(allsb, 50. - 68.3 / 2., axis=1)
    pmch = np.percentile(allsb, 50. + 68.3 / 2., axis=1)
    Mhyd.sb = pmc
    Mhyd.sb_lo = pmcl
    Mhyd.sb_hi = pmch

    alldens = np.sqrt(np.dot(Mhyd.Kdens, np.exp(Mhyd.samples.T)) / Mhyd.ccf * Mhyd.transf)
    pmc = np.median(alldens, axis=1)
    pmcl = np.percentile(alldens, 50. - 68.3 / 2., axis=1)
    pmch = np.percentile(alldens, 50. + 68.3 / 2., axis=1)
    Mhyd.dens = pmc
    Mhyd.dens_lo = pmcl
    Mhyd.dens_hi = pmch

    if Mhyd.spec_data is not None:
        kt_mod = kt_from_samples(Mhyd, mod, nmore=Mhyd.nmore)
        Mhyd.ktmod = kt_mod['TSPEC']
        Mhyd.ktmod_lo = kt_mod['TSPEC_LO']
        Mhyd.ktmod_hi = kt_mod['TSPEC_HI']
        Mhyd.kt3d = kt_mod['T3D']
        Mhyd.kt3d_lo = kt_mod['T3D_LO']
        Mhyd.kt3d_hi = kt_mod['T3D_HI']

    if Mhyd.sz_data is not None:
        pmed, plo, phi = P_from_samples(Mhyd, mod, nmore=Mhyd.nmore)
        Mhyd.pmod = pmed
        Mhyd.pmod_lo = plo
        Mhyd.pmod_hi = phi

    return mod

def SaveGP(Mhyd, outfile=None):
    if outfile is None:
        outfile = Mhyd.dir + '/output_GP.fits'

    if Mhyd.samples is None:
        print('Nothing to save, exiting')

        return

    if Mhyd.ngauss is None:
        print('No GP reconstruction found in structure, exiting')
        return

    hdus = fits.HDUList(hdus=[fits.PrimaryHDU()])

    denshdu = fits.ImageHDU(Mhyd.samples, name='DENSITY')

    headsamp = denshdu.header

    headsamp['BKGLIM'] = Mhyd.bkglim
    headsamp['NRC'] = Mhyd.nrc
    headsamp['NBETAS'] = Mhyd.nbetas
    headsamp['MINBETA'] = Mhyd.min_beta
    headsamp['CCF'] = Mhyd.ccf
    headsamp['TRANSF'] = Mhyd.transf
    headsamp['FITBKG'] = Mhyd.fit_bkg
    headsamp['NMORE'] = Mhyd.nmore

    hdus.append(denshdu)

    gphdu = fits.ImageHDU(Mhyd.samppar, name='GP')
    modhead = gphdu.header
    modhead['SMIN'] = Mhyd.smin
    modhead['SMAX'] = Mhyd.smax
    modhead['BINFACT'] = Mhyd.bin_fact
    modhead['NGAUSS'] = Mhyd.ngauss

    hdus.append(gphdu)

    hdus.writeto(outfile, overwrite=True)


def ReloadGP(Mhyd, infile):
    fin = fits.open(infile)

    Mhyd.samples = fin[1].data

    headden = fin[1].header

    Mhyd.bkglim = headden['BKGLIM']

    Mhyd.nrc = headden['NRC']

    Mhyd.nbetas = headden['NBETAS']

    Mhyd.min_beta = headden['MINBETA']

    Mhyd.ccf = headden['CCF']

    Mhyd.transf = headden['TRANSF']

    Mhyd.fit_bkg = headden['FITBKG']

    Mhyd.nmore = headden['NMORE']

    Mhyd.samppar = fin[2].data

    modhead = fin[2].header

    Mhyd.smin = modhead['SMIN']

    Mhyd.smax = modhead['SMAX']

    Mhyd.bin_fact = modhead['BINFACT']

    Mhyd.ngauss = modhead['NGAUSS']

    # Now recreate operators

    prof = Mhyd.sbprof

    rad = prof.bins

    area = prof.area

    exposure = prof.effexp

    sourcereg = np.where(rad < Mhyd.bkglim)

    pars = list_params(rad, sourcereg, Mhyd.nrc, Mhyd.nbetas, Mhyd.min_beta)

    npt = len(pars)

    if prof.psfmat is not None:
        psfmat = np.transpose(prof.psfmat)
    else:
        psfmat = np.eye(prof.nbin)

    Mhyd.pardens = list_params_density(rad, sourcereg, Mhyd.amin2kpc, Mhyd.nrc, Mhyd.nbetas, Mhyd.min_beta)

    # Compute linear combination kernel
    if Mhyd.fit_bkg:

        Mhyd.K = calc_linear_operator(rad, sourcereg, pars, area, exposure,
                                                psfmat)  # transformation to counts
        Mhyd.Kdens = calc_density_operator(rad, Mhyd.pardens, Mhyd.amin2kpc)

    else:

        Mhyd.K = calc_sb_operator_psf(rad, sourcereg, pars, area, exposure,
                                                psfmat)  # transformation to surface brightness
        Mhyd.Kdens = calc_density_operator(rad, Mhyd.pardens, Mhyd.amin2kpc, withbkg=False)

    # Define the fine grid onto which the mass model will be computed
    rin_m, rout_m, index_x, index_sz, sum_mat = rads_more(Mhyd, nmore=Mhyd.nmore)

    if Mhyd.fit_bkg:

        Mhyd.Kdens_m = calc_density_operator(rout_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc)

        Mhyd.Kdens_grad = calc_grad_operator(rout_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc)

    else:

        Mhyd.Kdens_m = calc_density_operator(rout_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc,
                                                       withbkg=False)

        Mhyd.Kdens_grad = calc_grad_operator(rout_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc,
                                                       withbkg=False)

    if Mhyd.fit_bkg:

        Ksb = calc_sb_operator(rad, sourcereg, pars)

        allsb = np.dot(Ksb, np.exp(Mhyd.samples.T))

        bfit = np.median(np.exp(Mhyd.samples[:, npt]))

        Mhyd.bkg = bfit

    else:

        Ksb = calc_sb_operator(rad, sourcereg, pars, withbkg=False)

        allsb = np.dot(Ksb, np.exp(Mhyd.samples.T))

    if Mhyd.spec_data is not None and Mhyd.sz_data is None:

        rout_joint = Mhyd.spec_data.rout_x

    elif Mhyd.spec_data is None and Mhyd.sz_data is not None:

        rout_joint = Mhyd.sz_data.rout_sz

    elif Mhyd.spec_data is not None and Mhyd.sz_data is not None:

        rout_joint = np.sort(np.append(Mhyd.spec_data.rout_x, Mhyd.sz_data.rout_sz))

    rin_joint = np.roll(rout_joint, 1)

    rin_joint[0] = 0.

    Mhyd.GPop, rgauss, sig = calc_gp_operator(Mhyd.ngauss, rout_m, rin_joint, rout_joint,
                                                        bin_fact=Mhyd.bin_fact, smin=Mhyd.smin, smax=Mhyd.smax)

    Mhyd.GPgrad = calc_gp_grad_operator(Mhyd.ngauss, rout_m, rin_joint, rout_joint, bin_fact=Mhyd.bin_fact,
                                                  smin=Mhyd.smin, smax=Mhyd.smax)

    pmc = np.median(allsb, axis=1)
    pmcl = np.percentile(allsb, 50. - 68.3 / 2., axis=1)
    pmch = np.percentile(allsb, 50. + 68.3 / 2., axis=1)
    Mhyd.sb = pmc
    Mhyd.sb_lo = pmcl
    Mhyd.sb_hi = pmch

    alldens = np.sqrt(np.dot(Mhyd.Kdens, np.exp(Mhyd.samples.T)) / Mhyd.ccf * Mhyd.transf)
    pmc = np.median(alldens, axis=1)
    pmcl = np.percentile(alldens, 50. - 68.3 / 2., axis=1)
    pmch = np.percentile(alldens, 50. + 68.3 / 2., axis=1)
    Mhyd.dens = pmc
    Mhyd.dens_lo = pmcl
    Mhyd.dens_hi = pmch

    if Mhyd.spec_data is not None:
        kt_mod = kt_GP_from_samples(Mhyd, nmore=Mhyd.nmore)
        Mhyd.ktmod = kt_mod['TSPEC']
        Mhyd.ktmod_lo = kt_mod['TSPEC_LO']
        Mhyd.ktmod_hi = kt_mod['TSPEC_HI']
        Mhyd.kt3d = kt_mod['T3D']
        Mhyd.kt3d_lo = kt_mod['T3D_LO']
        Mhyd.kt3d_hi = kt_mod['T3D_HI']

    if Mhyd.sz_data is not None:
        pmed, plo, phi = P_GP_from_samples(Mhyd, nmore=Mhyd.nmore)
        Mhyd.pmod = pmed
        Mhyd.pmod_lo = plo
        Mhyd.pmod_hi = phi


def SaveForward(Mhyd, Forward, outfile=None):
    if outfile is None:
        outfile = Mhyd.dir + '/output_forward.fits'

    if Mhyd.samples is None:
        print('Nothing to save, exiting')

        return

    if Forward is None:
        print('No Forward model provided, exiting')
        return

    hdus = fits.HDUList(hdus=[fits.PrimaryHDU()])

    denshdu = fits.ImageHDU(Mhyd.samples, name='DENSITY')

    headsamp = denshdu.header

    headsamp['BKGLIM'] = Mhyd.bkglim
    headsamp['NRC'] = Mhyd.nrc
    headsamp['NBETAS'] = Mhyd.nbetas
    headsamp['MINBETA'] = Mhyd.min_beta
    headsamp['CCF'] = Mhyd.ccf
    headsamp['TRANSF'] = Mhyd.transf
    headsamp['FITBKG'] = Mhyd.fit_bkg
    headsamp['NMORE'] = Mhyd.nmore

    hdus.append(denshdu)

    cols = []
    for i in range(Forward.npar):
        col = fits.Column(name=Forward.parnames[i], format='E', array=Mhyd.samppar[:, i])

        cols.append(col)

    coldefs = fits.ColDefs(cols)

    modhdu = fits.BinTableHDU.from_columns(coldefs)

    modhdu.name = 'FORWARD MODEL'

    modhead = modhdu.header

    for i in range(Forward.npar):
        parname = Forward.parnames[i]

        modhead['ST' + parname] = Forward.start[i]

        modhead['SD' + parname] = Forward.sd[i]

        modhead['LOW' + parname] = Forward.limits[i][0]

        modhead['HI' + parname] = Forward.limits[i][1]

    hdus.append(modhdu)

    hdus.writeto(outfile, overwrite=True)

def ReloadForward(Mhyd, infile):
    fin = fits.open(infile)

    Mhyd.samples = fin[1].data

    headden = fin[1].header

    Mhyd.bkglim = headden['BKGLIM']

    Mhyd.nrc = headden['NRC']

    Mhyd.nbetas = headden['NBETAS']

    Mhyd.min_beta = headden['MINBETA']

    Mhyd.ccf = headden['CCF']

    Mhyd.transf = headden['TRANSF']

    Mhyd.fit_bkg = headden['FITBKG']

    Mhyd.nmore = headden['NMORE']

    modhead = fin[2].header

    mod = Forward()

    for i in range(mod.npar):
        parname = mod.parnames[i]

        mod.start[i] = modhead['ST' + parname]

        mod.sd[i] = modhead['SD' + parname]

        mod.limits[i][0] = modhead['LOW' + parname]

        mod.limits[i][1] = modhead['HI' + parname]

    nsamp = int(fin[2].header['NAXIS2'])

    Mhyd.samppar = np.empty((nsamp, mod.npar))

    dpar = fin[2].data

    for i in range(mod.npar):
        name = mod.parnames[i]

        Mhyd.samppar[:, i] = dpar[name]

    # Now recreate operators

    prof = Mhyd.sbprof

    rad = prof.bins

    area = prof.area

    exposure = prof.effexp

    sourcereg = np.where(rad < Mhyd.bkglim)

    pars = list_params(rad, sourcereg, Mhyd.nrc, Mhyd.nbetas, Mhyd.min_beta)

    npt = len(pars)

    if prof.psfmat is not None:
        psfmat = np.transpose(prof.psfmat)
    else:
        psfmat = np.eye(prof.nbin)

    Mhyd.pardens = list_params_density(rad, sourcereg, Mhyd.amin2kpc, Mhyd.nrc, Mhyd.nbetas, Mhyd.min_beta)

    # Compute linear combination kernel
    if Mhyd.fit_bkg:

        Mhyd.K = calc_linear_operator(rad, sourcereg, pars, area, exposure,
                                                psfmat)  # transformation to counts
        Mhyd.Kdens = calc_density_operator(rad, Mhyd.pardens, Mhyd.amin2kpc)

    else:

        Mhyd.K = calc_sb_operator_psf(rad, sourcereg, pars, area, exposure,
                                                psfmat)  # transformation to surface brightness
        Mhyd.Kdens = calc_density_operator(rad, Mhyd.pardens, Mhyd.amin2kpc, withbkg=False)

    # Define the fine grid onto which the mass model will be computed
    rin_m, rout_m, index_x, index_sz, sum_mat = rads_more(Mhyd, nmore=Mhyd.nmore)

    if Mhyd.fit_bkg:

        Mhyd.Kdens_m = calc_density_operator(rout_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc)

    else:

        Mhyd.Kdens_m = calc_density_operator(rout_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc,
                                                       withbkg=False)

    if Mhyd.fit_bkg:

        Ksb = calc_sb_operator(rad, sourcereg, pars)

        allsb = np.dot(Ksb, np.exp(Mhyd.samples.T))

        bfit = np.median(np.exp(Mhyd.samples[:, npt]))

        Mhyd.bkg = bfit

    else:

        Ksb = calc_sb_operator(rad, sourcereg, pars, withbkg=False)

        allsb = np.dot(Ksb, np.exp(Mhyd.samples.T))

    pmc = np.median(allsb, axis=1)
    pmcl = np.percentile(allsb, 50. - 68.3 / 2., axis=1)
    pmch = np.percentile(allsb, 50. + 68.3 / 2., axis=1)
    Mhyd.sb = pmc
    Mhyd.sb_lo = pmcl
    Mhyd.sb_hi = pmch

    alldens = np.sqrt(np.dot(Mhyd.Kdens, np.exp(Mhyd.samples.T)) / Mhyd.ccf * Mhyd.transf)
    pmc = np.median(alldens, axis=1)
    pmcl = np.percentile(alldens, 50. - 68.3 / 2., axis=1)
    pmch = np.percentile(alldens, 50. + 68.3 / 2., axis=1)
    Mhyd.dens = pmc
    Mhyd.dens_lo = pmcl
    Mhyd.dens_hi = pmch

    if Mhyd.spec_data is not None:
        kt_mod = kt_forw_from_samples(Mhyd, mod, nmore=Mhyd.nmore)
        Mhyd.ktmod = kt_mod['TSPEC']
        Mhyd.ktmod_lo = kt_mod['TSPEC_LO']
        Mhyd.ktmod_hi = kt_mod['TSPEC_HI']
        Mhyd.kt3d = kt_mod['T3D']
        Mhyd.kt3d_lo = kt_mod['T3D_LO']
        Mhyd.kt3d_hi = kt_mod['T3D_HI']

    if Mhyd.sz_data is not None:
        pmed, plo, phi = P_forw_from_samples(Mhyd, mod, nmore=Mhyd.nmore)
        Mhyd.pmod = pmed
        Mhyd.pmod_lo = plo
        Mhyd.pmod_hi = phi

    return mod

