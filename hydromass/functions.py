import numpy as np
import scipy.special as special
import pymc as pm
import aesara as theano
import aesara.tensor as tt


class ArcTan(tt.Op):
    '''
    Theano arctan class
    '''
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def perform(self, node, inputs, outputs):
        x, = inputs
        y = np.arctan(x)
        outputs[0][0] = np.array(y)

    def grad(self, inputs, g):
        x, = inputs
        return [g[0] / (1 + x ** 2)]


tt_arctan = ArcTan()


def f_ein_mu(x, mu):
    '''
    Einasto mass model as a function of :math:`\\mu=1/\\alpha` (Mamon & Lokas 2005)

    :param x: Scaled radius
    :type x: float
    :param mu: Einasto mu
    :type mu: float
    :return: Scaled mass profile
    :rtype: float
    '''
    n0 = 2. * mu

    n1 = 3. * mu

    fx = mu ** (1. - n1) / 2. ** n1 * np.exp(n0) * special.gamma(n1) * special.gammainc(n1, n0 * x ** (1. / mu))

    return fx


def f_ein_mu_der(x, mu):
    '''
    Analytical derivative of the Einasto mass profile

    :param x: Scaled radius
    :type x: float
    :param mu: Einasto mu
    :type mu: float
    :return: Derivative of the scaled mass profile
    :rtype: float
    '''
    n0 = 2. * mu

    fx = x ** 2 * np.exp(n0) * np.exp(- n0 * x ** (1. / mu))

    return fx


# Class to implement the 2-parameter Einasto mass profile and its derivative
class EinastoFx(tt.Op):
    '''
    Theano class implementing the 2-parameter Einasto model with fixed mu=5.0

    '''
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def perform(self, node, inputs, outputs):
        x, = inputs
        y = f_ein_mu(x, 5.)
        outputs[0][0] = np.array(y)

    def grad(self, inputs, g):
        x, = inputs
        return [g[0] * f_ein_mu_der(x, 5.)]


tt_einasto = EinastoFx()


# 2-parameter Einasto function
def f_ein2_pm(xout, c200, r200, delta=200.):
    '''
    Theano function returning the 2-parameter Einasto model for a given set of parameters.

    .. math::

        \\rho(r) = \\rho_s \\exp \\left[ -2\\mu \\left( \\left( \\frac{r}{r_s} \\right) ^ {1 /\\mu} - 1 \\right) \\right]

    with :math:\\mu=5.0

    :param xout: Radius
    :param c200: concentration
    :param r200: Scale radius
    :param delta: Overdensity
    :return: Enclosed mass
    '''

    fcc = delta / 3. * c200 ** 3 / (pm.math.log(1. + c200) - c200 / (1. + c200))

    x = xout / r200

    fx = tt_einasto(x)  # To be updated to free mu

    return r200 ** 3 * fcc * fx


# 3-parameter Einasto function, we need to integrate the density numerically since the gradient is not analytical
def f_ein3_pm(xout, c200, r200, mu, delta=200.):
    '''
    Theano function for the 3-parameter Einasto mass model,

    .. math::

        M(r) = f_c R_{\\Delta}^3 \\int_{0}^{r} 4 \\pi r^2 \\rho(r) dr

    with

    .. math::

        \\rho(x) = \\exp \\left[ -2\\mu \\left( \\left( \\frac{r}{R_{\\Delta}} \\right) ^ {1 /\\mu} - 1 \\right) \\right]

    and

    .. math::

        f_c = \\frac{\\Delta}{3} \\frac{c^3}{\\ln(1+c)- c/(1+c)}

    :param xout: Radius
    :param c200: concentration
    :param r200: Scale radius
    :param mu: Einasto mu
    :param delta: Overdensity
    :return: Enclosed mass

    '''
    fcc = delta / 3. * c200 ** 3 / (pm.math.log(1. + c200) - c200 / (1. + c200))

    npt = len(xout)

    dr = (xout - np.roll(xout, 1))

    dr[0] = xout[0]

    x = (xout - dr / 2.) / r200

    n0 = 2. * mu

    integrand = x ** 2 * pm.math.exp(n0) * pm.math.exp(- n0 * x ** (1. / mu)) * dr / r200

    matrad = np.ones((npt, npt))

    matrad_tril = np.tril(matrad)

    fx = pm.math.dot(matrad_tril, integrand)

    return r200 ** 3 * fcc * fx


def f_ein2_np(xout, pars, delta=200.):
    '''
    Numpy function for the Einasto 2-parameter model

    .. math::

        \\rho(r) = \\rho_s \\exp \\left[ -2\\mu \\left( \\left( \\frac{r}{r_s} \\right) ^ {1 /\\mu} - 1 \\right) \\right]

    with :math:\\mu=5.0

    :param xout: Radius
    :type xout: numpy.ndarray
    :param pars: Model parameters (cdelta and rs)
    :type pars: numpy.ndarray
    :param delta: Overdensity
    :type delta: float
    :return: Enclosed mass
    :rtype: numpy.ndarray
    '''
    c200 = pars[:, 0]

    r200 = pars[:, 1]

    nn = 5

    n0 = 2. * nn

    n1 = 3. * nn

    npt = len(xout)

    npars = len(c200)

    c200mul = np.repeat(c200, npt).reshape(npars, npt)

    r200mul = np.repeat(r200, npt).reshape(npars, npt)

    fcc = delta / 3. * c200mul ** 3 / (np.log(1. + c200mul) - c200mul / (1. + c200mul))

    xoutmul = np.tile(xout, npars).reshape(npars, npt)

    x = xoutmul / r200mul

    fx = nn ** (1. - n1) / 2. ** n1 * np.exp(n0) * special.gamma(n1) * special.gammainc(n1, n0 * x ** (1. / nn))

    return r200mul ** 3 * fcc * fx


def f_ein3_np(xout, pars, delta=200.):
    '''
    Numpy function for the numerically integrated 3-parameter Einasto mass model (see :func:`hydromass.functions.f_ein3_pm`)

    :param xout: Radius
    :type xout: numpy.ndarray
    :param pars: 2D array with the chains of parameters of the mass model (cdelta, rs, and mu)
    :type pars: numpy.ndarray
    :param delta: Overdensity
    :type delta: float
    :return: Enclosed mass
    :rtype: numpy.ndarray
    '''
    c200 = pars[:, 0]

    r200 = pars[:, 1]

    mu = pars[:, 2]

    npt = len(xout)

    npars = len(c200)

    c200mul = np.repeat(c200, npt).reshape(npars, npt)

    r200mul = np.repeat(r200, npt).reshape(npars, npt)

    mumul = np.repeat(mu, npt).reshape(npars, npt)

    n0 = 2. * mumul

    n1 = 3. * mumul

    fcc = delta / 3. * c200mul ** 3 / (np.log(1. + c200mul) - c200mul / (1. + c200mul))

    xoutmul = np.tile(xout, npars).reshape(npars, npt)

    x = xoutmul / r200mul

    fx = mumul ** (1. - n1) / 2. ** n1 * np.exp(n0) * special.gamma(n1) * special.gammainc(n1, n0 * x ** (1. / mumul))

    return r200mul ** 3 * fcc * fx


# Isothermal sphere function
def f_iso_pm(xout, c200, r200, delta=200.):
    '''
    Theano function for the cored isothermal sphere mass profile (King 1962)

    .. math::

        M(r) = f_c R_{\\Delta}^3 \\left[ \\ln(x + \\sqrt{1+x^2}) - \\frac{x}{\\sqrt{1+x^2}} \\right]

    with :math:`x=r/R_{\\Delta}` and

    .. math::

        f_c = \\frac{\\Delta}{3} \\frac{c^3}{\\ln(1+c)- c/(1+c)}

    :param xout: Radius
    :param c200: concentration
    :param r200: Scale radius
    :param delta: Overdensity
    :return: Enclosed mass
    '''

    fcc = delta / 3. * c200 ** 3 / (pm.math.log(1. + c200) - c200 / (1. + c200))

    x = xout / r200

    fx = pm.math.log(x + pm.math.sqrt(1. + x ** 2)) - x / pm.math.sqrt(1. + x ** 2)

    return r200 ** 3 * fcc * fx


def f_iso_np(xout, pars, delta=200.):
    '''
    Numpy function for the cored isothermal sphere mass profile (see :func:`hydromass.functions.f_iso_pm`)

    :param xout: Radius
    :type xout: numpy.ndarray
    :param pars: 2D array with the chains of parameters of the mass model (cdelta, rs, and mu)
    :type pars: numpy.ndarray
    :param delta: Overdensity
    :type delta: float
    :return: Enclosed mass
    :rtype: numpy.ndarray
    '''

    c200 = pars[:, 0]

    r200 = pars[:, 1]

    npt = len(xout)

    npars = len(c200)

    c200mul = np.repeat(c200, npt).reshape(npars, npt)

    r200mul = np.repeat(r200, npt).reshape(npars, npt)

    fcc = delta / 3. * c200mul ** 3 / (np.log(1. + c200mul) - c200mul / (1. + c200mul))

    xoutmul = np.tile(xout, npars).reshape(npars, npt)

    x = xoutmul / r200mul

    fx = np.log(x + np.sqrt(1. + x ** 2)) - x / np.sqrt(1. + x ** 2)

    return r200mul ** 3 * fcc * fx


# NFW function



def f_nfw_pm(xout, c200, r200, delta=200.):
    '''
    Theano function for the Navarro-Frenk-White mass profile (Navarro et al. 1996)

    .. math::

        M(r) = f_c R_{\\Delta}^3 \\left[ \\ln(1 + x) - \\frac{x}{1+x}\\right]

    with :math:`x=r/R_{\\Delta}` and

    .. math::

        f_c = \\frac{\\Delta}{3} \\frac{c^3}{\\ln(1+c)- c/(1+c)}

    :param xout: Radius
    :param c200: concentration
    :param r200: Scale radius
    :param delta: Overdensity
    :return: Enclosed mass
    '''

    fcc = delta / 3. * c200 ** 3 / (pm.math.log(1. + c200) - c200 / (1. + c200))

    x = xout / r200 * c200

    fx = pm.math.log(1. + x) - x / (1. + x)

    return r200 ** 3 * fcc / c200 ** 3 * fx

def rho_nfw_cr(radii, c200, r200, delta=200.):
    # Theano function for the Navarro-Frank-White density profile (Navarro et al. 1996)
    # Should be multiplied by rho_crit(z)
    r = (radii[1:] + radii[:-1]) / 2 * 1000.
    delta_crit = (delta / 3) * (c200 ** 3) * (pm.math.log(1. + c200) - c200 / (1 + c200)) ** (-1)
    return delta_crit / ((c200 * r / r200) * ((1. + (c200 * r / r200)) ** 2))

def rho_nfw_cr_np(radii, c200, r200, delta=200.):
    # Theano function for the Navarro-Frank-White density profile (Navarro et al. 1996)
    # Should be multiplied by rho_crit(z)
    r = (radii[1:] + radii[:-1]) / 2 * 1000.
    delta_crit = (delta / 3) * (c200 ** 3) * (np.log(1. + c200) - c200 / (1 + c200)) ** (-1)
    return delta_crit / ((c200 * r / r200) * ((1. + (c200 * r / r200)) ** 2))

def f_nfw_np(xout, pars, delta=200.):
    '''
    Numpy function for the NFW profile (see :func:`hydromass.functions.f_nfw_pm`)

    :param xout: Radius
    :type xout: numpy.ndarray
    :param pars: 2D array with the chains of parameters of the mass model (cdelta, rs, and mu)
    :type pars: numpy.ndarray
    :param delta: Overdensity
    :type delta: float
    :return: Enclosed mass
    :rtype: numpy.ndarray
    '''
    c200 = pars[:, 0]

    r200 = pars[:, 1]

    npt = len(xout)

    npars = len(c200)

    c200mul = np.repeat(c200, npt).reshape(npars, npt)

    r200mul = np.repeat(r200, npt).reshape(npars, npt)

    fcc = delta / 3. * c200mul ** 3 / (np.log(1. + c200mul) - c200mul / (1. + c200mul))

    xoutmul = np.tile(xout, npars).reshape(npars, npt)

    x = xoutmul / r200mul * c200mul

    fx = np.log(1. + x) - x / (1. + x)

    return r200mul ** 3 * fcc / c200mul ** 3 * fx


# Hernquist function
def f_her_pm(xout, c200, r200, delta=200.):
    '''
    Theano function for the Hernquist (1990) mass profile,

    .. math::

        M(r) = f_c R_{\\Delta}^3 \\frac{x^2}{(x+1)^2}

    with :math:`x=r/R_{\\Delta}` and

    .. math::

        f_c = \\frac{\\Delta}{3} \\frac{c^3}{\\ln(1+c)- c/(1+c)}

    :param xout: Radius
    :param c200: concentration
    :param r200: Scale radius
    :param delta: Overdensity
    :return: Enclosed mass
    '''
    fcc = delta / 3. * c200 ** 3 / (pm.math.log(1. + c200) - c200 / (1. + c200))

    x = xout / r200

    fx = x ** 2. / (x + 1.) ** 2

    return r200 ** 3 * fcc * fx


def f_her_np(xout, pars, delta=200.):
    '''
    Numpy function for the Hernquist (1990) mass profile (see :func:`hydromass.functions.f_her_pm`)

    :param xout: Radius
    :type xout: numpy.ndarray
    :param pars: 2D array with the chains of parameters of the mass model (cdelta, rs, and mu)
    :type pars: numpy.ndarray
    :param delta: Overdensity
    :type delta: float
    :return: Enclosed mass
    :rtype: numpy.ndarray
    '''
    c200 = pars[:, 0]

    r200 = pars[:, 1]

    npt = len(xout)

    npars = len(c200)

    c200mul = np.repeat(c200, npt).reshape(npars, npt)

    r200mul = np.repeat(r200, npt).reshape(npars, npt)

    fcc = delta / 3. * c200mul ** 3 / (np.log(1. + c200mul) - c200mul / (1. + c200mul))

    xoutmul = np.tile(xout, npars).reshape(npars, npt)

    x = xoutmul / r200mul

    fx = x ** 2. / (x + 1.) ** 2

    return r200mul ** 3 * fcc * fx


# Burkert function
def f_bur_pm(xout, c200, r200, delta=200.):
    '''
    Theano function for the Burkert mass profile (Salucci & Burkert 2000),

    .. math::

        M(r) = f_c R_{\\Delta}^3 \\left[ \\ln(1+x^2) + 2\\ln(1+x) -2\\arctan(x) \\right]

    with :math:`x=r/R_{\\Delta}` and

    .. math::

        f_c = \\frac{\\Delta}{3} \\frac{c^3}{\\ln(1+c)- c/(1+c)}

    :param xout: Radius
    :param c200: concentration
    :param r200: Scale radius
    :param delta: Overdensity
    :return: Enclosed mass
    '''
    fcc = delta / 3. * c200 ** 3 / (pm.math.log(1. + c200) - c200 / (1. + c200))

    x = xout / r200

    fx = pm.math.log(1. + x ** 2) + 2. * pm.math.log(1. + x) - 2. * tt_arctan(x)

    return r200 ** 3 * fcc * fx


def f_bur_np(xout, pars, delta=200.):
    '''
    Numpy function for the Burkert mass profile (Salucci & Burkert 2000), see :func:`hydromass.functions.f_bur_pm`

    :param xout: Radius
    :type xout: numpy.ndarray
    :param pars: 2D array with the chains of parameters of the mass model (cdelta, rs, and mu)
    :type pars: numpy.ndarray
    :param delta: Overdensity
    :type delta: float
    :return: Enclosed mass
    :rtype: numpy.ndarray
    '''
    c200 = pars[:, 0]

    r200 = pars[:, 1]

    npt = len(xout)

    npars = len(c200)

    c200mul = np.repeat(c200, npt).reshape(npars, npt)

    r200mul = np.repeat(r200, npt).reshape(npars, npt)

    fcc = delta / 3. * c200mul ** 3 / (np.log(1. + c200mul) - c200mul / (1. + c200mul))

    xoutmul = np.tile(xout, npars).reshape(npars, npt)

    x = xoutmul / r200mul

    fx = np.log(1. + x ** 2) + 2. * np.log(1. + x) - 2. * np.arctan(x)

    return r200mul ** 3 * fcc * fx


class Model:
    '''
    Class defining mass models to be passed to the hydromass :class:`hydromass.mhyd.Mhyd` class for optimization.

    :param massmod:
        Name of the chosen mass model. Currently available mass models are:
            - 'NFW': Navarro-Frenk-White (1996) model, :func:`hydromass.functions.f_nfw_pm`
            - 'EIN2': Analytic 2-parameter Einasto (1965) model with :math:`\\alpha=0.2` (Mamon & Lokas 2005), :func:`hydromass.functions.f_ein2_pm`
            - 'EIN3': Numerically integrated 3-parameter Einasto (1965) model with free :math:`\alpha`, :func:`hydromass.functions.f_ein3_pm`
            - 'HER': Hernquist (1990) model, :func:`hydromass.functions.f_her_pm`
            - 'ISO': Cored isothermal sphere model (King 1962), :func:`hydromass.functions.f_iso_pm`
            - 'BUR': Salucci & Burkert (2000) model, :func:`hydromass.functions.f_bur_pm`
    :type massmod: str
    :param delta: Chosen fit overdensity. Defaults to 200
    :type delta: float
    :param start: 1D array containing the central values of the Gaussian prior on the model parameters. If None, the priors are set automatically to "sensible" values (for the galaxy cluster case...)
    :type start: numpy.ndarray
    :param sd: 1D array containing the standard deviations of the Gaussian prior on the model parameters. If None, weak priors are set automatically to remain within "sensible" values
    :type sd: numpy.ndarray
    :param limits: 2D array including the upper and lower boundaries of the parameter values. If None, the boundaries are set automatically to remain within "sensible" values
    :type limits: numpy.ndarray
    :param fix: 1D array setting whether each parameter of the mass model is fitted (False) or fixed (True). If None, all the parameters are free to vary.
    :type fix: numpy.ndarray
    '''

    def __init__(self, massmod, delta=200., start=None, sd=None, limits=None, fix=None):

        if massmod == 'NFW':

            func_pm = f_nfw_pm

            func_np = f_nfw_np

            self.rho_pm = rho_nfw_cr
            self.rho_np = rho_nfw_cr_np

            self.npar = 2
            self.parnames = ['cdelta', 'rdelta']

            if start is None:
                self.start = [4., 2000.]
            else:
                try:
                    assert (len(start) == self.npar)
                except AssertionError:
                    print('Number of starting parameters does not match function.')
                    return

                self.start = start

            if sd is None:

                self.sd = [2., 500.]

            else:

                try:
                    assert (len(sd) == self.npar)
                except AssertionError:
                    print('Shape of sd does not match function.')
                    return

                self.sd = sd

            if limits is None:

                limits = np.empty((self.npar, 2))

                limits[0] = [0., 15.]

                limits[1] = [300., 3000.]

            else:

                try:
                    assert (limits.shape == (self.npar, 2))
                except AssertionError:
                    print('Shape of limits does not match function.')
                    return

            if fix is None:

                self.fix = [False, False]

            else:

                try:
                    assert (len(fix) == self.npar)
                except AssertionError:
                    print('Shape of fix vectory does not match function.')
                    return

                self.fix = fix


        elif massmod == 'ISO':

            func_pm = f_iso_pm

            func_np = f_iso_np

            self.npar = 2
            self.parnames = ['cdelta', 'rdelta']

            if start is None:
                self.start = [7., 440.]
            else:
                try:
                    assert (len(start) == self.npar)
                except AssertionError:
                    print('Number of starting parameters does not match function.')
                    return

                self.start = start

            if sd is None:

                self.sd = [4., 200.]

            else:

                try:
                    assert (len(sd) == self.npar)
                except AssertionError:
                    print('Shape of sd does not match function.')
                    return

                self.sd = sd

            if limits is None:

                limits = np.empty((self.npar, 2))

                limits[0] = [1., 30.]

                limits[1] = [50., 2000.]

            else:

                try:
                    assert (limits.shape == (self.npar, 2))

                except AssertionError:

                    print('Shape of limits does not match function.')

                    return

            if fix is None:

                self.fix = [False, False]

            else:

                try:
                    assert (len(fix) == self.npar)
                except AssertionError:
                    print('Shape of fix vectory does not match function.')
                    return

                self.fix = fix

        elif massmod == 'EIN2':

            func_pm = f_ein2_pm

            func_np = f_ein2_np

            self.npar = 2
            self.parnames = ['cdelta', 'rdelta']

            if start is None:
                self.start = [1.8, 700.]
            else:
                try:
                    assert (len(start) == self.npar)
                except AssertionError:
                    print('Number of starting parameters does not match function.')
                    return

                self.start = start

            if sd is None:

                self.sd = [1.5, 300.]

            else:

                try:
                    assert (len(sd) == self.npar)
                except AssertionError:
                    print('Shape of sd does not match function.')
                    return

                self.sd = sd

            if limits is None:

                limits = np.empty((self.npar, 2))

                limits[0] = [0., 5.]

                limits[1] = [100., 1800.]

            else:

                try:
                    assert (limits.shape == (self.npar, 2))

                except AssertionError:

                    print('Shape of limits does not match function.')

                    return

            if fix is None:

                self.fix = [False, False]

            else:

                try:
                    assert (len(fix) == self.npar)
                except AssertionError:
                    print('Shape of fix vectory does not match function.')
                    return

                self.fix = fix


        elif massmod == 'EIN3':

            func_pm = f_ein3_pm

            func_np = f_ein3_np

            self.npar = 3

            self.parnames = ['cdelta', 'rdelta', 'mu']

            if start is None:
                self.start = [1.8, 700., 5.]
            else:
                try:
                    assert (len(start) == self.npar)
                except AssertionError:
                    print('Number of starting parameters does not match function.')
                    return

                self.start = start

            if sd is None:

                self.sd = [1.5, 300., 3.]

            else:

                try:
                    assert (len(sd) == self.npar)
                except AssertionError:
                    print('Shape of sd does not match function.')
                    return

                self.sd = sd

            if limits is None:

                limits = np.empty((self.npar, 2))

                limits[0] = [0., 5.]

                limits[1] = [100., 1800.]

                limits[2] = [1., 20.]

            else:

                try:
                    assert (limits.shape == (self.npar, 2))

                except AssertionError:

                    print('Shape of limits does not match function.')

                    return

            if fix is None:

                self.fix = [False, False, False]

            else:

                try:
                    assert (len(fix) == self.npar)
                except AssertionError:
                    print('Shape of fix vectory does not match function.')
                    return

                self.fix = fix


        elif massmod == 'BUR':

            func_pm = f_bur_pm

            func_np = f_bur_np

            self.npar = 2
            self.parnames = ['cdelta', 'rdelta']

            if start is None:
                self.start = [4.5, 200.]
            else:
                try:
                    assert (len(start) == self.npar)
                except AssertionError:
                    print('Number of starting parameters does not match function.')
                    return

                self.start = start

            if sd is None:

                self.sd = [3., 100.]

            else:

                try:
                    assert (len(sd) == self.npar)
                except AssertionError:
                    print('Shape of sd does not match function.')
                    return

                self.sd = sd

            if limits is None:

                limits = np.empty((self.npar, 2))

                limits[0] = [0., 20.]

                limits[1] = [20., 800.]

            else:

                try:
                    assert (limits.shape == (self.npar, 2))

                except AssertionError:

                    print('Shape of limits does not match function.')

                    return

            if fix is None:

                self.fix = [False, False]

            else:

                try:
                    assert (len(fix) == self.npar)
                except AssertionError:
                    print('Shape of fix vectory does not match function.')
                    return

                self.fix = fix


        elif massmod == 'HER':

            func_pm = f_her_pm

            func_np = f_her_np

            self.npar = 2

            self.parnames = ['cdelta', 'rdelta']

            if start is None:
                self.start = [1.9, 1000.]
            else:
                try:
                    assert (len(start) == self.npar)
                except AssertionError:
                    print('Number of starting parameters does not match function.')
                    return

                self.start = start

            if sd is None:

                self.sd = [2., 400.]

            else:

                try:
                    assert (len(sd) == self.npar)
                except AssertionError:
                    print('Shape of sd does not match function.')
                    return

                self.sd = sd

            if limits is None:

                limits = np.empty((self.npar, 2))

                limits[0] = [0., 6.]

                limits[1] = [200., 2500.]

            else:

                try:
                    assert (limits.shape == (self.npar, 2))

                except AssertionError:

                    print('Shape of limits does not match function.')

                    return

            if fix is None:

                self.fix = [False, False]

            else:

                try:
                    assert (len(fix) == self.npar)
                except AssertionError:
                    print('Shape of fix vectory does not match function.')
                    return

                self.fix = fix


        else:
            print('Error: Unknown mass model %s . Available mass models: NFW, ISO, EIN2, EIN3, BUR, HER.' % (massmod))

            return

        self.delta = delta

        self.massmod = massmod

        self.func_pm = func_pm

        self.func_np = func_np

        self.limits = limits

    # def __call__(self, x, pars):
    #
    # 	return self.func_pm(x, *pars)

    def __call__(self, x, pars):

        return self.func_np(x, *pars, delta=self.delta)
