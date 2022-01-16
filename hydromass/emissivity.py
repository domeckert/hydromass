import os

def is_tool(name):
    """Check whether `name` is on PATH."""

    from distutils.spawn import find_executable

    return find_executable(name) is not None


def calc_emissivity(cosmo, z, nh, kt, rmf, abund='angr', Z=0.3, elow=0.5, ehigh=2.0, arf=None):
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
    :return: Conversion factor
    :rtype: float
    """

    check_xspec = is_tool('xspec')

    if not check_xspec:

        print('Error: XSPEC cannot be found in path')

        return

    H0 = cosmo.H0.value

    Ode = cosmo.Ode0

    fsim=open('commands.xcm','w')

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

    fsim.write('\n')

    fsim.write('10000, 1\n')

    fsim.write('ign **-%1.2lf\n' % (elow))

    fsim.write('ign %1.2lf-**\n' % (ehigh))

    fsim.write('log sim.txt\n')

    fsim.write('show rates\n')

    fsim.write('log none\n')

    fsim.write('quit\n')

    fsim.close()

    nrmf_tot = rmf.split('.')[0]

    if os.path.exists('%s.fak' % (nrmf_tot)):

        os.system('rm %s.fak' % (nrmf_tot))

    srmf = nrmf_tot.split('/')

    nrmf = srmf[len(srmf) - 1]

    if os.path.exists('%s.fak' % (nrmf)):

        os.system('rm %s.fak' % (nrmf))

    os.system('xspec < commands.xcm')

    ssim = os.popen('grep cts/s sim.txt','r')

    lsim = ssim.readline()

    cr = float(lsim.split()[6])

    #ccf = 1. / cr

    return cr




