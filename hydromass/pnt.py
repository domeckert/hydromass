import pkg_resources
import os
import numpy as np
import pymc as pm

def r200m_from_params(c, z):
    '''
    Approximate the ratio of :math:`R_{200m}` to :math:`R_{200c}` as a function of NFW concentration and redshift

    :param c: NFW concentration
    :type c: float
    :param z: Source redshift
    :type z: float
    :return: Ratio
    :rtype: float
    '''

    R0 = 1.79

    alpha = -0.06

    beta = -0.64

    ratio = R0 * c ** alpha * (1. + z) ** beta

    return ratio


def alpha_turb_pm(rad, r200c, c200, z, pars):
    '''
    Theano function implementing the model non-thermal pressure fraction following Angelinelli et al. (2020),

    .. math::

        \\frac{P_{NT}}{P_{TOT}} = a_0 \\left(\\frac{r}{R_{200c}}\\right)^{a_1} + a_2

    :param rad: Radius
    :param r200c: Value of :math:`R_{200c}`
    :param c200: NFW concentration
    :param z: Redshift
    :param pars: Non-thermal pressure model parameters
    :return: Non-thermal pressure fraction
    '''

    a0 = pars[0,0]

    a1 = pars[0,1]

    a2 = pars[0,2]

    r200m = r200c * r200m_from_params(c200, z)

    xm = rad / r200m

    pnt = a0 * xm ** a1 + a2

    return pnt


def alpha_turb_np(rad, pars, z, pnt_pars):
    '''
    Numpy function for the non-thermal pressure fraction, see :func:`hydromass.pnt.alpha_turb_pm`

    :param rad: Radii
    :type rad: numpy.ndarray
    :param pars: Samples of NFW concentration and overdensity radii
    :type pars: numpy.ndarray
    :param z: Source redshift
    :type z: float
    :param pnt_pars:  Non-thermal pressure parameter samples
    :type pnt_pars: numpy.ndarray
    :return: Profiles of non-thermal pressure fraction
    :rtype: numpy.ndarray
    '''

    a0 = pnt_pars[:, 0]

    a1 = pnt_pars[:, 1]

    a2 = pnt_pars[:, 2]

    npt = len(rad)

    nobj = len(a0)

    c200 = pars[:, 0]

    r200 = pars[:, 1]

    r200mul = np.tile(r200, npt).reshape(npt, nobj)

    c200mul = np.tile(c200, npt).reshape(npt, nobj)

    rat_mul = r200m_from_params(c200mul, z)

    xmul = np.repeat(rad, nobj).reshape(npt, nobj) / r200mul / rat_mul # R/R_200m

    a0mul = np.tile(a0, npt).reshape(npt, nobj)

    a1mul = np.tile(a1, npt).reshape(npt, nobj)

    a2mul = np.tile(a2, npt).reshape(npt, nobj)

    return a0mul * xmul ** a1mul + a2mul


def get_data_file_path(data_file):
    """
    Returns the absolute path to the required data files.

    :param data_file: relative path to the data file, relative to the hydromass/data path.
    :return: absolute path of the data file
    """

    try:

        file_path = pkg_resources.resource_filename("hydromass", 'data/%s' % data_file)

    except KeyError:

        raise IOError("Could not read or find data file %s." % (data_file))

    else:

        return os.path.abspath(file_path)

