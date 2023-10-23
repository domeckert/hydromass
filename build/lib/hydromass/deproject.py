import numpy as np
from scipy.special import gamma
from .constants import *
from astropy.io import fits
import time

# Function to calculate a linear operator transforming parameter vector into predicted model counts

def calc_linear_operator(rad,sourcereg,pars,area,expo,psf):
    '''
    Function to calculate a linear operator transforming parameter vector into predicted model counts

    .. math::

        C(r) = \\sum_{i=1}^P \\alpha_i C_i(r)

    with :math:`\\alpha_i` the parameter values and :math:`C_i(r)` the count profiles of each basis function, i.e. the indices of the output matrix


    :param rad: Array of input radii in arcmin
    :type rad: numpy.ndarray
    :param sourcereg: Selection array for the source region
    :type sourcereg: numpy.ndarray
    :param pars: List of beta model parameters obtained through :func:`hydromass.deproject.list_params`
    :type pars: numpy.ndarray
    :param area: Bin area in arcmin^2
    :type area: numpy.ndarray
    :param expo: Bin effective exposure in s
    :type expo: numpy.ndarray
    :param psf: PSF mixing matrix
    :type psf: numpy.ndarray
    :return: Linear projection and PSF mixing operator
    :rtype: numpy.ndarray
    '''

    # Select values in the source region
    rfit=rad[sourcereg]
    npt=len(rfit)
    npars=len(pars[:,0])
    areamul=np.tile(area[0:npt],npars).reshape(npars,npt)
    expomul=np.tile(expo[0:npt],npars).reshape(npars,npt)
    spsf=psf[0:npt,0:npt]
    
    # Compute linear combination of basis functions in the source region
    beta=np.repeat(pars[:,0],npt).reshape(npars,npt)
    rc=np.repeat(pars[:,1],npt).reshape(npars,npt)
    base=1.+np.power(rfit/rc,2)
    expon=-3.*beta+0.5
    func_base=np.power(base,expon)
    
    # Predict number of counts per annulus and convolve with PSF
    Ktrue=func_base*areamul*expomul
    Kconv=np.dot(spsf,Ktrue.T)
    
    # Recast into full matrix and add column for background
    nptot=len(rad)
    Ktot=np.zeros((nptot,npars+1))
    Ktot[0:npt,0:npars]=Kconv
    Ktot[:,npars]=area*expo
    return Ktot


# Function to create the list of parameters for the basis functions
nsh=4. # number of basis functions to set

def list_params(rad,sourcereg,nrc=None,nbetas=6,min_beta=0.6):
    """
    Define a list of parameters to define the dictionary of basis functions

    :param rad: Array of input radii in arcmin
    :type rad: numpy.ndarray
    :param sourcereg: Selection array for the source region
    :type sourcereg: numpy.ndarray
    :param nrc: Number of core radii. If nrc=None (default), the number of core radiis will be defined adaptively as one per each set of 4 data points
    :type nrc: int
    :param nbetas: Number of beta values. Defaults to 6
    :type nbetas: int
    :param min_beta: Minimum value of beta. Defaults to 0.6
    :type min_beta: float
    :return: Array containing sets of values to set up the function dictionary
    :rtype: numpy.ndarray
    """
    rfit=rad[sourcereg]
    npfit=len(rfit)
    if nrc is None:
        nrc = np.max([int(npfit/nsh),1])
    allrc=np.logspace(np.log10(rfit[2]),np.log10(rfit[npfit-1]/2.),nrc)
    #allbetas=np.linspace(0.4,3.,6)
    allbetas = np.linspace(min_beta, 3., nbetas)
    nrc=len(allrc)
    nbetas=len(allbetas)
    rc=allrc.repeat(nbetas)
    betas=np.tile(allbetas,nrc)
    ptot=np.empty((nrc*nbetas,2))
    ptot[:,0]=betas
    ptot[:,1]=rc
    return ptot

# Function to create a linear operator transforming parameters into surface brightness

def calc_sb_operator(rad,sourcereg,pars, withbkg=True):
    """
    Function to calculate a linear operator transforming a parameter vector into a model surface brightness profile

    .. math::

        S_X(r) = \\sum_{i=1}^P \\alpha_i S_i(r)

    with :math:`\\alpha_i` the parameter values and :math:`S_i(r)` the brightness profiles of each basis functions, i.e. the indices of the output matrix

    :param rad: Array of input radii in arcmin
    :type rad: numpy.ndarray
    :param sourcereg: Selection array for the source region
    :type sourcereg: numpy.ndarray
    :param pars: List of beta model parameters obtained through list_params
    :type pars: numpy.ndarray
    :param withbkg: Set whether the background is fitted jointly (True) or subtracted (False). Defaults to True.
    :type withbkg: bool
    :return: Linear projection operator
    :rtype: numpy.ndarray
    """

    # Select values in the source region
    rfit=rad[sourcereg]
    npt=len(rfit)
    npars=len(pars[:,0])
    
    # Compute linear combination of basis functions in the source region
    beta=np.repeat(pars[:,0],npt).reshape(npars,npt)
    rc=np.repeat(pars[:,1],npt).reshape(npars,npt)
    base=1.+np.power(rfit/rc,2)
    expon=-3.*beta+0.5
    func_base=np.power(base,expon)
    
    # Recast into full matrix and add column for background
    if withbkg:
        nptot=len(rad)
        Ktot=np.zeros((nptot,npars+1))
        Ktot[0:npt,0:npars]=func_base.T
        Ktot[:,npars]=0.0

    else:
        Ktot = func_base.T

    return Ktot


def calc_sb_operator_psf(rad, sourcereg, pars, area, expo, psf, withbkg=False):
    """
    Same as :func:`hydromass.deproject.calc_sb_operator` but convolving the model surface brightness with the PSF model

    :param rad: Array of input radii in arcmin
    :type rad: numpy.ndarray
    :param sourcereg: Selection array for the source region
    :type sourcereg: numpy.ndarray
    :param pars: List of beta model parameters obtained through list_params
    :type pars: numpy.ndarray
    :param area: Bin area in arcmin^2
    :type area: numpy.ndarray
    :param expo: Bin effective exposure in s
    :type expo: numpy.ndarray
    :param psf: PSF mixing matrix
    :type psf: numpy.ndarray
    :param withbkg: Set whether the background is fitted jointly (True) or subtracted (False). Defaults to False.
    :type withbkg: bool
    :return: Linear projection and PSF mixing operator
    :rtype: numpy.ndarray
    """

    # Select values in the source region
    rfit = rad[sourcereg]
    npt = len(rfit)
    npars = len(pars[:, 0])

    areamul = np.tile(area[0:npt], npars).reshape(npars, npt)
    expomul = np.tile(expo[0:npt], npars).reshape(npars, npt)
    spsf = psf[0:npt, 0:npt]

    # Compute linear combination of basis functions in the source region
    beta = np.repeat(pars[:, 0], npt).reshape(npars, npt)
    rc = np.repeat(pars[:, 1], npt).reshape(npars, npt)
    base = 1. + np.power(rfit / rc, 2)
    expon = -3. * beta + 0.5
    func_base = np.power(base, expon)

    Ktrue = func_base * areamul * expomul
    Kconv = np.dot(spsf, Ktrue.T)
    Kconvsb = Kconv / areamul.T / expomul.T

    # Recast into full matrix and add column for background
    if withbkg:
        nptot = len(rad)
        Ktot = np.zeros((nptot, npars + 1))
        Ktot[0:npt, 0:npars] = Kconvsb
        Ktot[:, npars] = 0.

    else:
        Ktot = Kconvsb

    return Ktot


def calc_int_operator(a, b, pars):
    """
    Compute a linear operator to integrate analytically the basis functions within some radial range and return count rate and luminosities

    .. math::

        CR = \\sum_{i=1}^P \\alpha_i (F_i(b) - F_i(a))

    with a,b the inner and outer radii of the chosen radial range, :math:`\\alpha_i` the parameter values and :math:`F_i(a), F_i(b)` the analytic integral of the basis functions, i.e. the indices of the output matrix

    :param a: Lower integration boundary
    :type a: float
    :param b: Upper integration boundary
    :type b: float
    :param pars: List of beta model parameters obtained through list_params
    :type pars: numpy.ndarray
    :return: Linear integration operator
    :rtype: numpy.ndarray
    """
    # Select values in the source region
    npars = len(pars[:, 0])
    rads = np.array([a, b])
    npt = 2

    # Compute linear combination of basis functions in the source region
    beta = np.repeat(pars[:, 0], npt).reshape(npars, npt)
    rc = np.repeat(pars[:, 1], npt).reshape(npars, npt)
    base = 1. + np.power(rads / rc, 2)
    expon = -3. * beta + 1.5
    func_base = 2. * np.pi * np.power(base, expon) / (3 - 6 * beta) * rc**2

    # Recast into full matrix and add column for background
    Kint = np.zeros((npt, npars + 1))
    Kint[0:npt, 0:npars] = func_base.T
    Kint[:, npars] = 0.0
    return Kint


def list_params_density(rad, sourcereg, kpcp, nrc=None, nbetas=6, min_beta=0.6):
    """
    Define a list of parameters to transform the basis functions into gas density profiles

    :param rad: Array of input radii in arcmin
    :type rad: numpy.ndarray
    :param sourcereg: Selection array for the source region
    :type sourcereg: numpy.ndarray
    :param z: Source redshift
    :type z: float
    :param nrc: Number of core radii. If nrc=None (default), the number of core radiis will be defined adaptively as one per each set of 4 data points.
    :type nrc: int
    :param nbetas: Number of beta values. Defaults to 6
    :type nbetas: int
    :param min_beta: Minimum value of beta. Defaults to 0.6
    :type min_beta: float
    :return: Array containing sets of values to set up the function dictionary
    :rtype: numpy.ndarray
    """
    rfit = rad[sourcereg]
    npfit = len(rfit)
    if nrc is None:
        nrc = np.max([int(npfit / nsh), 1])
    allrc = np.logspace(np.log10(rfit[2]), np.log10(rfit[npfit - 1] / 2.), nrc) * kpcp
    # allbetas=np.linspace(0.5,3.,6)
    allbetas = np.linspace(min_beta, 3., nbetas)
    nrc = len(allrc)
    nbetas = len(allbetas)
    rc = allrc.repeat(nbetas)
    betas = np.tile(allbetas, nrc)
    ptot = np.empty((nrc * nbetas, 2))
    ptot[:, 0] = betas
    ptot[:, 1] = rc
    return ptot


# Linear operator to transform parameters into density

def calc_density_operator(rad, pars, kpcp, withbkg=True):
    """
    Compute linear operator to transform a parameter vector into a gas density profile

    .. math::

        n_e(r) = \\sum_{i=1}^P \\alpha_i f_i(r)

    with :math:`\\alpha_i` the parameter values and :math:`f_i(r)` the profiles of each basis function, i.e. the indices of the output matrix

    :param rad: Array of input radii in arcmin
    :type rad: numpy.ndarray
    :param pars: List of beta model parameters obtained through :func:`hydromass.deproject.list_params_density`
    :type pars: numpy.ndarray
    :param kpcp: Kiloparsec equivalent of 1 arcmin at the redshift of the source
    :type kpcp: float
    :param withbkg: Set whether the background is fitted jointly (True) or subtracted (False). Defaults to True.
    :type withbkg: bool
    :return: Linear operator for gas density
    :rtype: numpy.ndarray
    """
    # Select values in the source region
    rfit = rad * kpcp
    npt = len(rfit)
    npars = len(pars[:, 0])

    # Compute linear combination of basis functions in the source region
    beta = np.repeat(pars[:, 0], npt).reshape(npars, npt)
    rc = np.repeat(pars[:, 1], npt).reshape(npars, npt)
    base = 1. + np.power(rfit / rc, 2)
    expon = -3. * beta
    func_base = np.power(base, expon)
    cfact = gamma(3 * beta) / gamma(3 * beta - 0.5) / np.sqrt(np.pi) / rc
    fng = func_base * cfact

    # Recast into full matrix and add column for background
    if withbkg:
        nptot=len(rfit)
        Ktot=np.zeros((nptot,npars+1))
        Ktot[0:npt,0:npars]=fng.T
        Ktot[:,npars]=0.0

    else:
        Ktot = fng.T

    return Ktot

# Function to compute d(log n)/d(log r)
def calc_grad_operator(rad, pars, kpcp, withbkg=True):
    '''
    Compute a linear operator transforming a parameter vector into a density gradient profile

    .. math::

        \\frac{\\partial \\log n_e}{\\partial \\log r} = \\sum_{i=1}^P \\alpha_i g_i(r)

    with :math:`\\alpha_i` the parameter values and :math:`g_i(r)` the log gradients of each basis functions, i.e. the indices of the output matrix

    :param rad: Array of input radii in arcmin
    :type rad: numpy.ndarray
    :param pars: List of beta model parameters obtained through :func:`hydromass.deproject.list_params_density`
    :type pars: numpy.ndarray
    :param kpcp: Kiloparsec equivalent of 1 arcmin at the redshift of the source
    :type kpcp: float
    :param withbkg: Set whether the background is fitted jointly (True) or subtracted (False). Defaults to True.
    :type withbkg: bool
    :return: Linear operator for gas density
    :rtype: numpy.ndarray
    '''
    # Select values in the source region
    rfit = rad * kpcp
    npt = len(rfit)
    npars = len(pars[:, 0])

    # Compute linear combination of basis functions in the source region
    beta = np.repeat(pars[:, 0], npt).reshape(npars, npt)
    rc = np.repeat(pars[:, 1], npt).reshape(npars, npt)
    base = 1. + np.power(rfit / rc, 2)
    expon = -3. * beta
    func_base = np.power(base, expon)
    cfact = gamma(3 * beta) / gamma(3 * beta - 0.5) / np.sqrt(np.pi) / rc
    n2 = func_base * cfact
    dlogn2dlogr = - 6. * beta * (rfit / rc) ** 2 / base
    grad = n2 * dlogn2dlogr

    # Recast into full matrix and add column for background
    if withbkg:
        nptot=len(rfit)
        Ktot=np.zeros((nptot,npars+1))
        Ktot[0:npt,0:npars]=grad.T
        Ktot[:,npars]=0.0

    else:
        Ktot = grad.T

    return Ktot


class MyDeprojVol:
    """
    Compute the projection volumes in spherical symmetry following Kriss et al. (1983)

    :param radin: Array of inner radii of the bins
    :type radin: class:`numpy.ndarray`
    :param radout: Array of outer radii of the bins
    :type radout: class:`numpy.ndarray`
    """
    def __init__(self, radin, radot):
        '''

        :param radin:

        :param radot:
        '''
        self.radin=radin
        self.radot=radot
        self.help=''

    def deproj_vol(self):
        """
        Compute the projection volumes

        :return: Volume matrix
        :rtype: numpy.ndarray
        """
        ###############volume=deproj_vol(radin,radot)
        ri=np.copy(self.radin)
        ro=np.copy(self.radot)

        diftot=0
        for i in range(1,len(ri)):
            dif=abs(ri[i]-ro[i-1])/ro[i-1]*100.
            diftot=diftot+dif
            ro[i-1]=ri[i]

        if abs(diftot) > 0.1:
            print(' DEPROJ_VOL: WARNING - abs(ri(i)-ro(i-1)) differs by',diftot,' percent')
            print(' DEPROJ_VOL: Fixing up radii ... ')
            for i in range(1,len(ri)-1):
                dif=abs(ri[i]-ro[i-1])/ro[i-1]*100.
                diftot=diftot+dif
        nbin=len(ro)
        volconst=4./3.*np.pi
        volmat=np.zeros((nbin, nbin))

        for iring in list(reversed(range(0,nbin))):
            volmat[iring,iring]=volconst * ro[iring]**3 * (1.-(ri[iring]/ro[iring])**2.)**1.5
            for ishell in list(reversed(range(iring+1,nbin))):
                f1=(1.-(ri[iring]/ro[ishell])**2.)**1.5 - (1.-(ro[iring]/ro[ishell])**2.)**1.5
                f2=(1.-(ri[iring]/ri[ishell])**2.)**1.5 - (1.-(ro[iring]/ri[ishell])**2.)**1.5
                volmat[ishell,iring]=volconst * (f1*ro[ishell]**3 - f2*ri[ishell]**3)

                if volmat[ishell,iring] < 0.0:
                    exit()

        volume2=np.copy(volmat)
        return volume2

