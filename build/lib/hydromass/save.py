from .plots import *
from .functions import *
from .nonparametric import *
from .forward import *

def SaveModel(Mhyd, model, outfile=None):
    '''
    Save the output chains of a fitted mass model to an output file. The output file can then be reloaded into a new Mhyd object using the :func:`hydromass.save.ReloadModel` function.

    :param Mhyd: A :class:`hydromass.mhyd.Mhyd` object containing the definition of the fitted data and and the output of the mass model fit
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param model: A :class:`hydromass.functions.Model` object containing the definition of the mass model
    :type model: :class:`hydromass.functions.Model`
    :param outfile: Name of output FITS file. If none, the file is outputted to a file called "output_model.fits" under the default output directory specified in the :class:`hydromass.mhyd.Mhyd` object. Defaults to none
    :type outfile: str
    '''
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
    headsamp['TRANSF'] = Mhyd.transf
    headsamp['FITBKG'] = Mhyd.fit_bkg
    headsamp['NMORE'] = Mhyd.nmore
    headsamp['DMONLY'] = Mhyd.dmonly
    headsamp['PNT'] = Mhyd.pnt
    if Mhyd.pnt:
        headsamp['PNTMODEL'] = Mhyd.pnt_model

    is_elong = False

    try:
        nn = len(Mhyd.elong)

    except TypeError:

        headsamp['ELONG'] = is_elong

    else:

        is_elong = True
        headsamp['ELONG'] = is_elong

    hdus.append(denshdu)

    cols = []
    for i in range(model.npar):
        col = fits.Column(name=model.parnames[i], format='E', array=Mhyd.samppar[:, i])

        cols.append(col)

    col = fits.Column(name='logP0', format='E', array=Mhyd.samplogp0)

    cols.append(col)

    # col = fits.Column(name='LogLike', format='E', array=Mhyd.totlike)
    #
    # cols.append(col)
    #
    # col = fits.Column(name='LogLikeTH', format='E', array=Mhyd.thermolike)
    #
    # cols.append(col)

    coldefs = fits.ColDefs(cols)

    modhdu = fits.BinTableHDU.from_columns(coldefs)

    modhdu.name = 'MASS MODEL'

    modhead = modhdu.header
    modhead['MASSMOD'] = model.massmod

    modhead['DELTA'] = model.delta

    # modhead['WAIC'] = Mhyd.waic['waic']
    #
    # modhead['LOO'] = Mhyd.loo['loo']

    for i in range(model.npar):
        parname = model.parnames[i]

        modhead['ST' + parname] = model.start[i]

        modhead['SD' + parname] = model.sd[i]

        modhead['LOW' + parname] = model.limits[i][0]

        modhead['HI' + parname] = model.limits[i][1]

    hdus.append(modhdu)

    bins = Mhyd.sbprof.bins

    colr = fits.Column(name='RADIUS', format='E', unit='arcmin', array=bins)

    ccfprof = None

    try:
        nn = len(Mhyd.ccf)

    except TypeError:

        ccfprof = np.ones(len(bins)) * Mhyd.ccf

    else:

        ccfprof = Mhyd.ccf

    colc = fits.Column(name='CCF', format='E', array=ccfprof)

    coldefs = fits.ColDefs([colr, colc])

    ccfhdu = fits.BinTableHDU.from_columns(coldefs)

    ccfhdu.name = 'CCF'

    hdus.append(ccfhdu)

    if Mhyd.pnt:

        nthdu = fits.ImageHDU(Mhyd.pnt_pars, name='NT')

        hdus.append(nthdu)

    if is_elong:

        elonghdu = fits.ImageHDU(Mhyd.elong, name='ELONG')

        hdus.append(elonghdu)

    hdus.writeto(outfile, overwrite=True)


def ReloadModel(Mhyd, infile, mstar=None):
    '''
    Reload the result of a previous mass model fit within the current live session after it has been saved using the :func:`hydromass.save.SaveModel` function

    :param Mhyd: A :class:`hydromass.mhyd.Mhyd` object into which the previously saved data will be stored
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param infile: Name of input FITS file containing the data saved using the :func:`hydromass.save.SaveModel` function
    :type infile: str
    :param mstar: External cumulative stellar mass profile if available. If none, no stellar mass profile is used. Defaults to none
    :type mstar: class:`numpy.ndarray`
    :return: A :class:`hydromass.functions.Model` object with the model definition and parameters
    :rtype: class:`hydromass.functions.Model`
    '''
    fin = fits.open(infile)

    Mhyd.samples = fin[1].data

    headden = fin[1].header

    Mhyd.bkglim = headden['BKGLIM']

    Mhyd.nrc = headden['NRC']

    Mhyd.nbetas = headden['NBETAS']

    Mhyd.min_beta = headden['MINBETA']

    Mhyd.transf = headden['TRANSF']

    Mhyd.fit_bkg = headden['FITBKG']

    Mhyd.nmore = headden['NMORE']

    Mhyd.dmonly = headden['DMONLY']

    if 'ELONG' in headden:

        is_elong = headden['ELONG']

    else:

        is_elong = False

    Mhyd.mstar = mstar

    Mhyd.pnt = headden['PNT']

    tabccf = fin['CCF'].data

    Mhyd.ccf = tabccf['CCF']

    if is_elong:

        tabelong = fin['ELONG']

        Mhyd.elong = tabelong.data

    else:

        Mhyd.elong = 1

    if 'PNTMODEL' in headden:

        Mhyd.pnt_model = headden['PNTMODEL']

    #Mhyd.waic = headden['WAIC']

    #Mhyd.loo = headden['LOO']

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

    if Mhyd.pnt:

        Mhyd.pnt_pars = fin['NT'].data

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

        Ksb = calc_sb_operator(rad, sourcereg, pars, withbkg=False)

        Mhyd.K = np.dot(prof.psfmat, Ksb) # transformation to surface brightness

        Mhyd.Kdens = calc_density_operator(rad, Mhyd.pardens, Mhyd.amin2kpc, withbkg=False)

    # Define the fine grid onto which the mass model will be computed
    rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore=Mhyd.nmore)

    rref_m = (rin_m + rout_m)/2.

    cf = np.interp(rref_m, rad * Mhyd.amin2kpc, Mhyd.ccf)

    Mhyd.cf_prof = cf

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

        allsb_conv = np.dot(prof.psfmat, allsb[:, :npt])

    else:

        Ksb = calc_sb_operator(rad, sourcereg, pars, withbkg=False)

        if is_elong:

            elong_mat = np.tile(Mhyd.elong, Mhyd.sbprof.nbin).reshape(Mhyd.sbprof.nbin,nsamp)

            allsb = np.dot(Ksb, np.exp(Mhyd.samples.T)) * elong_mat ** 0.5

            allsb_conv = np.dot(Mhyd.K, np.exp(Mhyd.samples.T)) * elong_mat ** 0.5

        else:

            allsb = np.dot(Ksb, np.exp(Mhyd.samples.T))

            allsb_conv = np.dot(Mhyd.K, np.exp(Mhyd.samples.T))

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

    alldens = np.sqrt(np.dot(Mhyd.Kdens, np.exp(Mhyd.samples.T)) * Mhyd.transf)
    pmc = np.median(alldens, axis=1) / np.sqrt(Mhyd.ccf)
    pmcl = np.percentile(alldens, 50. - 68.3 / 2., axis=1) / np.sqrt(Mhyd.ccf)
    pmch = np.percentile(alldens, 50. + 68.3 / 2., axis=1) / np.sqrt(Mhyd.ccf)

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
    '''
    Save the result of a non-parametric GP reconstruction into an output FITS file. The result can be later reloaded through the :func:`hydromass.save.ReloadGP` function

    :param Mhyd:  A :class:`hydromass.mhyd.Mhyd` object containing the definition of the fitted data and and the output of the non-parametric GP reconstruction
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param outfile: Name of output FITS file. If none, the file is outputted to a file called "output_GP.fits" under the default output directory specified in the :class:`hydromass.mhyd.Mhyd` object. Defaults to none
    :type outfile: str
    '''
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

    bins = Mhyd.sbprof.bins

    colr = fits.Column(name='RADIUS', format='E', unit='arcmin', array=bins)

    ccfprof = None

    try:
        nn = len(Mhyd.ccf)

    except TypeError:

        ccfprof = np.ones(len(bins)) * Mhyd.ccf

    else:

        ccfprof = Mhyd.ccf

    colc = fits.Column(name='CCF', format='E', array=ccfprof)

    coldefs = fits.ColDefs([colr, colc])

    ccfhdu = fits.BinTableHDU.from_columns(coldefs)

    ccfhdu.name = 'CCF'

    hdus.append(ccfhdu)

    hdus.writeto(outfile, overwrite=True)


def ReloadGP(Mhyd, infile):
    '''
    Reload the result of a non-parametric GP reconstruction previously saved using the :func:`hydromass.save.SaveGP` function into the current session.

    :param Mhyd: A :class:`hydromass.mhyd.Mhyd` object into which the previously saved data will be stored
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param infile: Name of input FITS file containing the data saved using the :func:`hydromass.save.SaveGP` function
    :type infile: str
    '''
    fin = fits.open(infile)

    Mhyd.samples = fin[1].data

    headden = fin[1].header

    Mhyd.bkglim = headden['BKGLIM']

    Mhyd.nrc = headden['NRC']

    Mhyd.nbetas = headden['NBETAS']

    Mhyd.min_beta = headden['MINBETA']

    Mhyd.transf = headden['TRANSF']

    Mhyd.fit_bkg = headden['FITBKG']

    Mhyd.nmore = headden['NMORE']

    tabccf = fin['CCF'].data

    Mhyd.ccf = tabccf['CCF']

    Mhyd.samppar = fin[2].data

    modhead = fin[2].header

    Mhyd.smin = modhead['SMIN']

    Mhyd.smax = modhead['SMAX']

    Mhyd.bin_fact = modhead['BINFACT']

    Mhyd.ngauss = modhead['NGAUSS']

    Mhyd.cf_prof = None

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

    # Define the fine grid onto which the mass model will be computed
    rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore=Mhyd.nmore)

    # Compute linear combination kernel
    if Mhyd.fit_bkg:

        Mhyd.K = calc_linear_operator(rad, sourcereg, pars, area, exposure,
                                                psfmat)  # transformation to counts
        Mhyd.Kdens = calc_density_operator(rad, Mhyd.pardens, Mhyd.amin2kpc)

        Mhyd.Kdens_m = calc_density_operator(rout_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc)

        Mhyd.Kdens_grad = calc_grad_operator(rout_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc)
    else:

        Ksb = calc_sb_operator(rad, sourcereg, pars, withbkg=False)

        Mhyd.K = np.dot(prof.psfmat, Ksb)  # transformation to surface brightness

        Mhyd.Kdens = calc_density_operator(rad, Mhyd.pardens, Mhyd.amin2kpc, withbkg=False)

        Mhyd.Kdens_m = calc_density_operator(rout_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc,
                                             withbkg=False)
        Mhyd.Kdens_grad = calc_grad_operator(rout_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc,
                                             withbkg=False)

    if Mhyd.fit_bkg:

        Ksb = calc_sb_operator(rad, sourcereg, pars)

        allsb = np.dot(Ksb, np.exp(Mhyd.samples.T))

        bfit = np.median(np.exp(Mhyd.samples[:, npt]))

        Mhyd.bkg = bfit

        allsb_conv = np.dot(prof.psfmat, allsb[:, :npt])

    else:

        Ksb = calc_sb_operator(rad, sourcereg, pars, withbkg=False)

        allsb = np.dot(Ksb, np.exp(Mhyd.samples.T))

        allsb_conv = np.dot(Mhyd.K, np.exp(Mhyd.samples.T))

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

    if Mhyd.spec_data is not None and Mhyd.sz_data is None:

        rout_joint = Mhyd.spec_data.rout_x

    elif Mhyd.spec_data is None and Mhyd.sz_data is not None:

        rout_joint = Mhyd.sz_data.rout_sz

    elif Mhyd.spec_data is not None and Mhyd.sz_data is not None:

        rout_joint = np.sort(np.append(Mhyd.spec_data.rout_x, Mhyd.sz_data.rout_sz))

    rin_joint = np.roll(rout_joint, 1)

    rin_joint[0] = 0.

    Mhyd.GPop, rgauss, sig = calc_gp_operator_lognormal(Mhyd.ngauss, rout_m, rin_joint, rout_joint,
                                                        bin_fact=Mhyd.bin_fact, smin=Mhyd.smin, smax=Mhyd.smax)

    Mhyd.GPgrad = calc_gp_grad_operator_lognormal(Mhyd.ngauss, rout_m, rin_joint, rout_joint, bin_fact=Mhyd.bin_fact,
                                                  smin=Mhyd.smin, smax=Mhyd.smax)

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
    '''

    :param Mhyd: A :class:`hydromass.mhyd.Mhyd` object containing the definition of the fitted data and the output of the forward fit
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param Forward: A :class:`hydromass.forward.Forward` object containing the definition of the forward model
    :type Forward: class:`hydromass.forward.Forward`
    :param outfile: Name of output FITS file. If none, the file is outputted to a file called "output_forward.fits" under the default output directory specified in the :class:`hydromass.mhyd.Mhyd` object. Defaults to none
    :type outfile: str
    '''
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

    bins = Mhyd.sbprof.bins

    colr = fits.Column(name='RADIUS', format='E', unit='arcmin', array=bins)

    ccfprof = None

    try:
        nn = len(Mhyd.ccf)

    except TypeError:

        ccfprof = np.ones(len(bins)) * Mhyd.ccf

    else:

        ccfprof = Mhyd.ccf

    colc = fits.Column(name='CCF', format='E', array=ccfprof)

    coldefs = fits.ColDefs([colr, colc])

    ccfhdu = fits.BinTableHDU.from_columns(coldefs)

    ccfhdu.name = 'CCF'

    hdus.append(ccfhdu)

    hdus.writeto(outfile, overwrite=True)

def ReloadForward(Mhyd, infile):
    '''
    Reload the results of a previous forward fit saved using the :func:`hydromass.save.SaveForward` function into the current live session

    :param Mhyd: A :class:`hydromass.mhyd.Mhyd` object into which the previously saved data will be stored
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param infile: Name of input FITS file containing the data saved using the :func:`hydromass.save.SaveForward` function
    :type infile: str
    '''
    fin = fits.open(infile)

    Mhyd.samples = fin[1].data

    headden = fin[1].header

    Mhyd.bkglim = headden['BKGLIM']

    Mhyd.nrc = headden['NRC']

    Mhyd.nbetas = headden['NBETAS']

    Mhyd.min_beta = headden['MINBETA']

    Mhyd.transf = headden['TRANSF']

    Mhyd.fit_bkg = headden['FITBKG']

    Mhyd.nmore = headden['NMORE']

    tabccf = fin['CCF'].data

    Mhyd.ccf = tabccf['CCF']

    modhead = fin[2].header

    Mhyd.cf_prof = None

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

        Ksb = calc_sb_operator(rad, sourcereg, pars, withbkg=False)

        Mhyd.K = np.dot(prof.psfmat, Ksb)  # transformation to surface brightness

        Mhyd.Kdens = calc_density_operator(rad, Mhyd.pardens, Mhyd.amin2kpc, withbkg=False)

    # Define the fine grid onto which the mass model will be computed
    rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore=Mhyd.nmore)

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

        allsb_conv = np.dot(prof.psfmat, allsb[:, :npt])

    else:

        Ksb = calc_sb_operator(rad, sourcereg, pars, withbkg=False)

        allsb = np.dot(Ksb, np.exp(Mhyd.samples.T))

        allsb_conv = np.dot(Mhyd.K, np.exp(Mhyd.samples.T))

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

def SaveProfiles(profiles, outfile=None, extname='THERMODYNAMIC PROFILES'):
    '''
    Save the profiles loaded into a profile dictionary into an output FITS file

    :param profiles: Dictionary containing profiles and profile names
    :type profiles: dict
    :param outfile: Output FITS file name
    :type outfile: str
    :param extname: Name of the extension of the FITS file containing the output data. Defaults to "THERMODYNAMIC PROFILES"
    :type extname: str
    '''

    if outfile is None:

        print('Error: output file name must be provided')

        return

    hdus = fits.HDUList(hdus=[fits.PrimaryHDU()])

    cols = []

    for key, value in profiles.items():

        col = fits.Column(name=key, format='E', array=value)

        cols.append(col)

    coldefs = fits.ColDefs(cols)

    modhdu = fits.BinTableHDU.from_columns(coldefs)

    modhdu.name = extname

    hdus.append(modhdu)

    hdus.writeto(outfile, overwrite=True)

def LoadProfiles(infile):

    fitsfile = fits.open(infile)

    tabdata = fitsfile[1].data

    fitsfile.close()

    return tabdata