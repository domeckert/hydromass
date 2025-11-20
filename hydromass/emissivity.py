import os
import  numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize

def is_tool(name):
    """Check whether `name` is on PATH."""

    from distutils.spawn import find_executable

    return find_executable(name) is not None


def calc_emissivity(cosmo, z, nh, kt, rmf, abund='aspl', Z=0.3, elow=0.5, ehigh=2.0, unit='cr', lum_elow=0.5, lum_ehigh=2.0, arf=None, tmpdir='.', quiet=True):
    """

    Function calc_emissivity, computes scaling factor between count rate and APEC/MEKAL norm using XSPEC. The tool performs an XSPEC simulation of an absorbed APEC model with parameters set by the user, and computes the expected count rate in the band of interest corresponding to an APEC normalization of unity. The corresponding number is then used as conversion between count rate and emission measure for the hydrostatic mass reconstruction step. Note that since the computed count rates are corrected for vignetting through the exposure map, the provided ARF should correspond to the on-axis (unvignetted) effective area.
    Requires that XSPEC be in path

    :param cosmo: Astropy cosmology object containing the definition of the used cosmology.
    :type cosmo: class:`astropy.cosmology`
    :param z: Source redshift
    :type z: float
    :param nh: Source NH in units of 1e22 cm**(-2)
    :type nh: float
    :param kt: Source temperature in keV
    :type kt: float
    :param rmf: Path to response file (RMF/RSP)
    :type rmf: str
    :param abund: Solar abundance table in XSPEC format. Defaults to "aspl" (Asplund et al. 2009)
    :type abund: str
    :param Z: Metallicity with respect to solar. Defaults to 0.3
    :type Z: float
    :param elow: Low-energy bound of the input image in keV. Defaults to 0.5
    :type elow: float
    :param ehigh: High-energy bound of the input image in keV. Defaults to 2.0
    :type ehigh: float
    :param arf: Path to on-axis ARF (optional, in case response file is RMF)
    :type arf: str
    :param unit: Specify whether the exposure map is in units of sec (unit='cr') or photon flux (unit='photon'). By default unit='cr'.
    :type unit: str
    :param lum_elow: Low energy bound (rest frame) for luminosity calculation. Defaults to 0.5
    :type lum_elow: float
    :param lum_ehigh: High energy bound (rest frame) for luminosity calculation. Defaults to 2.0
    :type lum_ehigh: float
    :param quiet: Do not print all xspec output
    :type quiet: bool
    :return: Conversion factor
    :rtype: float
    """

    if unit!='cr' and unit!='photon':

        print('Unknown unit %s, aborting' % (unit))
        return

    check_xspec = is_tool('xspec')

    if not check_xspec:

        print('Error: XSPEC cannot be found in path')

        return

    if not os.path.exists(rmf):

        print('Error: RMF file not found, aborting')

        return

    if arf is not None and not os.path.exists(arf):

        print('Error: ARF file not found, aborting')

        return

    H0 = cosmo.H0.value

    Ode = cosmo.Ode0

    fakfile = tmpdir + '/sim.fak'

    if os.path.exists(fakfile):

        os.remove(fakfile)

    fsim=open(tmpdir + '/commands.xcm','w')

    # fsim.write('query y\n')

    fsim.write('cosmo %g 0 %g\n' % (H0, Ode))

    fsim.write('abund %s\n' % (abund))

    fsim.write('model phabs(apec)\n')

    fsim.write('%g\n'%(nh))

    fsim.write('%g\n'%(kt))

    fsim.write('%g\n'%(Z))

    fsim.write('%g\n'%(z))

    fsim.write('1.0\n')

    fsim.write('fakeit none\n')

    fsim.write('%s\n' % (rmf))

    if arf is not None:

        fsim.write('%s\n' % (arf))

    else:

        fsim.write('\n')

    fsim.write('\n')

    fsim.write('\n')

    fsim.write(f'{fakfile}\n')

    fsim.write('10000, 1\n')

    fsim.write('ign **-%1.2lf\n' % (elow))

    fsim.write('ign %1.2lf-**\n' % (ehigh))

    fsim.write(f'log {tmpdir}/sim.txt\n')

    if unit == 'cr':

        fsim.write('show rates\n')

        fsim.write('newpar 1 0 -1 0 0 1 1 \n')

        fsim.write('flux %1.2lf %1.2lf\n' % (elow, ehigh))


    elif unit == 'photon':

        fsim.write('flux %1.2lf %1.2lf\n' % (elow, ehigh))

    fsim.write('log none\n')

    fsim.write('delcomp 1\n')

    fsim.write(f'log {tmpdir}/lumin.txt\n')

    fsim.write('lumin %1.2lf %1.2lf %g\n' % (lum_elow, lum_ehigh, z))

    fsim.write('log none\n')

    fsim.write('quit\n')

    fsim.close()

    if quiet:

        os.system(f'xspec < {tmpdir}/commands.xcm > /dev/null 2>&1')

    else:

        os.system(f'xspec < {tmpdir}/commands.xcm')

    ssim = open(f'{tmpdir}/sim.txt')
    lsim = ssim.readlines()
    ssim.close()

    if unit == 'cr':

        for line in lsim:

            if 'Model predicted rate:' in line:

                cr = float(line.split()[4])

    else:

        for line in lsim:

            if 'photons' in line:

                cr = float(line.split()[3])

    for line in lsim:

        if 'Model Flux' in line:

            flux = float(line.split()[5].replace('(',''))

    slum = os.popen(f'grep Luminosity {tmpdir}/lumin.txt', 'r')

    llum = slum.readline()

    lumtot = float(llum.split()[2])

    lumfact = lumtot / cr

    return cr, lumfact, lumtot, cr, flux


def medsmooth(prof):
    """
    Smooth a given profile by taking the median value of surrounding points instead of the initial value

    :param prof: Input profile to be smoothed
    :type prof: numpy.ndarray
    :return: Smoothd profile
    :rtype: numpy.ndarray
    """
    width=5
    nbin=len(prof)
    xx=np.empty((nbin,width))
    xx[:,0]=np.roll(prof,2)
    xx[:,1]=np.roll(prof,1)
    xx[:,2]=prof
    xx[:,3]=np.roll(prof,-1)
    xx[:,4]=np.roll(prof,-2)
    smoothed=np.median(xx,axis=1)
    smoothed[1]=np.median(xx[1,1:width])
    smoothed[nbin-2]=np.median(xx[nbin-2,0:width-1])
    Y0=3.*prof[0]-2.*prof[1]
    xx=np.array([Y0,prof[0],prof[1]])
    smoothed[0]=np.median(xx)
    Y0=3.*prof[nbin-1]-2.*prof[nbin-2]
    xx=np.array([Y0,prof[nbin-2],prof[nbin-1]])
    smoothed[nbin-1]=np.median(xx)
    return  smoothed

def vikh_temp(x, pars, Tmin):
    T0=pars[0]
    #Tmin=pars[1]
    rcool=pars[1]
    acool=pars[2]
    rt=pars[3]
    c=pars[4]
    numer=Tmin/T0+np.power(x/rcool,acool)
    denom=1.+np.power(x/rcool,acool)
    t2=np.power(1.+(x/rt)**2,-c/2)
    return T0*numer/denom*t2

def variable_ccf(Mhyd, cosmo, z, nh, rmf, method='interp', abund='aspl', elow=0.5, ehigh=2.0,
                 unit='cr', lum_elow=0.5, lum_ehigh=2.0, arf=None, outz=None, outkt=None, tmpdir='.', out_cfact_file=None, quiet=False):
    '''

    :param Mhyd: Hydromass object
    :type Mhyd: :class:`hydromass.mhyd.Mhyd`
    :param cosmo: Astropy cosmology object containing the definition of the used cosmology.
    :type cosmo: class:`astropy.cosmology`
    :param z: Source redshift
    :type z: float
    :param nh: Source NH in units of 1e22 cm**(-2)
    :type nh: float
    :param rmf: Path to response file (RMF/RSP)
    :type rmf: str
    :param method: Choose whether the temperature profile will be interpolated (method='interp') or fitted with a parametric function (method='fit'). Defaults to 'interp'.
    :type method: str
    :param abund: Solar abundance table in XSPEC format. "aspl" (Asplund et al. 2009)
    :type abund: str
    :param elow: Low-energy bound of the input image in keV. Defaults to 0.5
    :type elow: float
    :param ehigh: High-energy bound of the input image in keV. Defaults to 2.0
    :type ehigh: float
    :param arf: Path to on-axis ARF (optional, in case response file is RMF)
    :type arf: str
    :param outz: Name of output file including the fit to the metal abundance profile. If None, it is ignored. Defaults to None.
    :type outz: str
    :param outkt: Name of output file including the fit to the temperature profile. If None, it is ignored. Defaults to None.
    :type outkt: str
    :param out_cfact_file: Output conversion factor file
    :type out_cfact_file: str
    :param quiet: Do not print all xspec output
    :type quiet: bool
    :return: Conversion factor
    :rtype: float
    '''

    if Mhyd.spec_data is None:

        print('No spectral data loaded, aborting')

        return

    spec_data = Mhyd.spec_data

    bins = Mhyd.sbprof.bins
    ebins = Mhyd.sbprof.ebins

    nbin = Mhyd.sbprof.nbin

    cf_prof, lf_prof = np.empty(nbin), np.empty(nbin)
    lum_prof, cr_prof, fx_prof = np.empty(nbin), np.empty(nbin), np.empty(nbin)

    nkt = len(spec_data.temp_x)

    if method=='fit':

        pars_temp = np.array([1.2, np.exp(-2.90), 1.06, 0.36, 0.28 * 2.0])

        def optim_kt(pars, Tmin):

            x = spec_data.rref_x / 500.  # assuming R500=500 kpc

            mod_kt = vikh_temp(x, pars, Tmin)

            chi2 = np.sum((mod_kt - spec_data.temp_x) ** 2 / spec_data.errt_x ** 2)

            if np.any(pars < 0.):

                chi2 = chi2 + 1e10

            return chi2

        Tmin = 0.6 * np.max(medsmooth(spec_data.temp_x))

        res = minimize(optim_kt, pars_temp, method='Nelder-Mead', args=(Tmin))

        ktfit = res['x']

        ktprof = vikh_temp(bins * Mhyd.amin2kpc / 500., ktfit, Tmin)

        if outkt is not None:

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

            plt.errorbar(spec_data.rref_x, spec_data.temp_x, xerr=(spec_data.rout_x - spec_data.rin_x) / 2.,
                         yerr=[spec_data.templ, spec_data.temph], fmt='o', label='Data')

            plt.plot(bins*Mhyd.amin2kpc, ktprof, color='green', label='Z profile')

            plt.xlabel('Radius [kpc]', fontsize=28)

            plt.ylabel('kT [keV]', fontsize=28)

            plt.savefig(outkt)


    else:

        if method != 'interp':
            print('Unknown method %s, reverting to interpolation' % (method))

        fill_value = (spec_data.temp_x[0], spec_data.temp_x[nkt - 1])

        fint = interp1d(spec_data.rref_x_am, medsmooth(spec_data.temp_x), kind='cubic', fill_value=fill_value,
                        bounds_error=False)


        ktprof = fint(bins)

    if spec_data.zfe is None:

        print('No abundance value loaded, assuming 0.3 everywhere')

        zfe_prof = np.empty(nbin)

        for i in range(nbin):



            cf_prof[i], lf_prof[i], lum_prof[i], cr_prof[i], fx_prof[i] = calc_emissivity(cosmo=cosmo,
                                         z=z,
                                         nh=nh,
                                         kt=ktprof[i],
                                         Z=0.3,
                                         elow=elow,
                                         ehigh=ehigh,
                                         rmf=rmf,
                                         abund=abund,
                                         arf=arf,
                                         unit=unit,
                                         lum_elow=lum_elow,
                                         lum_ehigh=lum_ehigh,
                                         tmpdir=tmpdir,
                                         quiet = quiet
                                         )

            zfe_prof[i] = 0.3

    else:

        print('Modeling abundance profile...')

        active = np.where(spec_data.zfe_lo > 0.)

        rads_z = spec_data.rref_x[active]

        beta_fe = 0.3

        pars_fe = np.array([30., 0.5, 0.2])

        def optim_zfe(pars):

            rc = pars[0]
            norm = pars[1]
            floor = pars[2]

            pred = floor + norm * (1. + (rads_z / rc) ** 2) ** (-beta_fe)

            chi2 = np.sum((pred - spec_data.zfe[active]) ** 2 / spec_data.zfe_hi[active] ** 2)

            if rc<=0 or floor<0 or norm<-0.2:

                chi2 = chi2 + 1e10

            return chi2

        res = minimize(optim_zfe, pars_fe, method='Nelder-Mead')

        zfit = res['x']

        med_rc = zfit[0]

        med_floor = zfit[2]

        med_norm = zfit[1]

        zfe_prof = med_floor + med_norm * (1. + (bins*Mhyd.amin2kpc/med_rc)**2) ** (-beta_fe)

        for i in range(nbin):

            cf_prof[i], lf_prof[i], lum_prof[i], cr_prof[i], fx_prof[i]  = calc_emissivity(cosmo=cosmo,
                                         z=z,
                                         nh=nh,
                                         kt=ktprof[i],
                                         Z=zfe_prof[i],
                                         elow=elow,
                                         ehigh=ehigh,
                                         rmf=rmf,
                                         abund=abund,
                                         arf=arf,
                                         unit=unit,
                                         lum_elow=lum_elow,
                                         lum_ehigh=lum_ehigh,
                                         quiet = quiet
                                         )

        if outz is not None:

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

            plt.errorbar(rads_z, spec_data.zfe[active], xerr=(spec_data.rout_x[active] - spec_data.rin_x[active]) / 2.,
                         yerr=[spec_data.zfe_lo[active], spec_data.zfe_hi[active]], fmt='o', label='Data')

            plt.plot(bins*Mhyd.amin2kpc, zfe_prof, color='green', label='Z profile')

            plt.xlabel('Radius [kpc]', fontsize=28)

            plt.ylabel('$Z/Z_{\odot}$', fontsize=28)

            plt.savefig(outz)
    if out_cfact_file:
        with open(out_cfact_file, 'w') as f:
            print('#           R           e_R            kT             Z            CR          FLUX           LUM ', file = f)
            for i in range(nbin):
                print(f'{bins[i]:13.7g} {ebins[i]:13.7g} {ktprof[i]:13.7g} {zfe_prof[i]:13.7g} {cr_prof[i]:13.7g} {fx_prof[i]:13.7g} {lum_prof[i]:13.7g}', file=f)


    return cf_prof, lf_prof
