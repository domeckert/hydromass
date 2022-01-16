import numpy as np
from scipy.special import gamma
from .constants import *
import pymc3 as pm
from astropy.io import fits
import time

# Function to calculate a linear operator transforming parameter vector into predicted model counts

def calc_linear_operator(rad,sourcereg,pars,area,expo,psf):
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
    '''
    Mydeproj
    '''
    def __init__(self, radin, radot):
        '''

        :param radin:

        :param radot:
        '''
        self.radin=radin
        self.radot=radot
        self.help=''

    def deproj_vol(self):
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

