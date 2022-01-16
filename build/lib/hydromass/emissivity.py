import os

def is_tool(name):
    """Check whether `name` is on PATH."""

    from distutils.spawn import find_executable

    return find_executable(name) is not None


def calc_emissivity(cosmo, z, nh, kt, rmf, Z=0.3, elow=0.5, ehigh=2.0, arf=None):
    """

    Function calc_emissivity, computes scaling factor between count rate and APEC/MEKAL norm using XSPEC.
    Requires that XSPEC be in path

    :param cosmo: (astropy.cosmology) Astropy cosmology object
    :param z: (float) Source redshift
    :param nh: (float) Source NH in units of 1e22 cm**(-2)
    :param kt: (float) Source temperature in keV
    :param rmf: (string) Path to response file (RMF/RSP)
    :param Z: (float) Metallicity with respect to solar (default = 0.3)
    :param elow: (float) Low-energy bound of the input image in keV (default = 0.5)
    :param ehigh: (float) High-energy bound of the input image in keV (default = 2.0)
    :param arf: (string) Path to on-axis ARF (optional, in case response file is RMF)
    :return: Conversion factor (float)
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

    fsim.write('abund angr\n')

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




