import numpy as np
import os
from astropy.io import fits
from .deproject import MyDeprojVol
from scipy.optimize import brentq
from scipy.signal import convolve


class SpecData:

    def __init__(self, redshift, spec_data=None, rin=None, rout=None, kt=None, err_kt_low=None, err_kt_high=None, cosmo=None):

        if spec_data is None and kt is None:

            print('No data provided. Please provide either an input FITS data file or a set of numpy arrays.')

            return

        if spec_data is not None and kt is not None:

            print('Ambiguous input. Please provide either an input FITS data file or a set of numpy arrays.')

            return

        if spec_data is not None:

            if not os.path.exists(spec_data):

                print('Spectral data file not found in path, skipping')

            else:

                print('Reading spectral data from file ' + spec_data)

                ftx = fits.open(spec_data)

                dtx = ftx[1].data

                cols = ftx[1].columns

                colnames = cols.names

                if 'RIN' in colnames:

                    rin = dtx['RIN']

                    rout = dtx['ROUT']

                elif 'RADIUS' in colnames:

                    rx = dtx['RADIUS']

                    erx = dtx['WIDTH']

                    rin = rx - erx

                    rout = rx + erx

                else:

                    print('No appropriate data found in input FITS table')

                    return

                self.temp_x = dtx['KT']

                self.templ = dtx['KT_LO']

                self.temph = dtx['KT_HI']

                ftx.close()

        if kt is not None:

            if rin is None or rout is None or err_kt_low is None or err_kt_high is None:

                print('Missing input, please provide rin, rout, kt, err_kt_low, and err_kt_high')

                return

            self.temp_x = kt

            self.templ = err_kt_low

            self.temph = err_kt_high

        if cosmo is None:

            from astropy.cosmology import Planck15 as cosmo

        amin2kpc = cosmo.kpc_proper_per_arcmin(redshift).value

        self.rin_x = rin * amin2kpc

        self.rout_x = rout * amin2kpc

        self.rin_x_am = rin

        self.rout_x_am = rout

        x = MyDeprojVol(rin, rout)

        self.volume = x.deproj_vol()

        self.errt_x = (self.temph + self.templ) / 2.

        self.rref_x = ((self.rin_x ** 1.5 + self.rout_x ** 1.5) / 2.) ** (2. / 3)

        self.rref_x_am = self.rref_x / amin2kpc

        self.psfmat = None



    def PSF(self, pixsize, psffunc=None, psffile=None, psfimage=None, psfpixsize=None, sourcemodel=None, psfmin=1e-7):
        #####################################################
        # Function to calculate a PSF convolution matrix given an input PSF image or function
        # Images of each annuli are convolved with the PSF image using FFT
        # FFT-convolved images are then used to determine PSF mixing
        #####################################################

        rad = (self.rin_x_am + self.rout_x_am) / 2.

        erad = (self.rout_x_am - self.rin_x_am) / 2.

        if psffile is None and psfimage is None and psffunc is None:

            print('No PSF model given')

            return

        else:
            if psffile is not None:

                fpsf = fits.open(psffile)

                psfimage = fpsf[0].data.astype(float)

                if psfpixsize is not None:

                    psfpixsize = float(psfimage[0].header['CDELT2'])

                    if psfpixsize == 0.0:

                        print('Error: no pixel size is provided for the PSF image and the CDELT2 keyword is not set')

                        return

                fpsf.close()

            if psfimage is not None:

                if psfpixsize is None or psfpixsize <= 0.0:

                    print('Error: no pixel size is provided for the PSF image')

                    return

            nbin = len(rad)

            psfout = np.zeros((nbin, nbin))

            npexp = 2 * int((rad[nbin - 1] + erad[nbin - 1]) / pixsize) + 1

            exposure = np.ones((npexp, npexp))

            y, x = np.indices(exposure.shape)

            rads = np.hypot(x - npexp / 2., y - npexp / 2.) * pixsize  # arcmin

            kernel = None

            if psffunc is not None:

                kernel = psffunc(rads)

                norm = np.sum(kernel)

                frmax = lambda x: psffunc(x) * 2. * np.pi * x / norm - psfmin

                if frmax(exposure.shape[0] / 2) < 0.:

                    rmax = brentq(frmax, 1., exposure.shape[0]) / pixsize  # pixsize

                    npix = int(rmax)

                else:
                    npix = int(exposure.shape[0] / 2)

                yp, xp = np.indices((2 * npix + 1, 2 * npix + 1))

                rpix = np.sqrt((xp - npix) ** 2 + (yp - npix) ** 2) * pixsize

                kernel = psffunc(rpix)

                norm = np.sum(kernel)

                kernel = kernel / norm

            if psfimage is not None:

                norm = np.sum(psfimage)

                kernel = psfimage / norm

            if kernel is None:

                print('No kernel provided, bye bye')

                return

            # Sort pixels into radial bins
            tol = 0.5e-5

            sort_list = []

            for n in range(nbin):

                if n == 0:

                    sort_list.append(np.where(np.logical_and(rads >= 0, rads < np.round(rad[n] + erad[n], 5) + tol)))

                else:

                    sort_list.append(np.where(np.logical_and(rads >= np.round(rad[n] - erad[n], 5) + tol,
                                                             rads < np.round(rad[n] + erad[n], 5) + tol)))

            # Calculate average of PSF image pixel-by-pixel and sort it by radial bins
            for n in range(nbin):

                # print('Working with bin',n+1)
                region = sort_list[n]

                npt = len(x[region])

                imgt = np.zeros(exposure.shape)

                if sourcemodel is None or sourcemodel.params is None:

                    imgt[region] = 1. / npt

                else:

                    imgt[region] = sourcemodel.model(rads[region], *sourcemodel.params)

                    norm = np.sum(imgt[region])

                    imgt[region] = imgt[region] / norm

                # FFT-convolve image with kernel
                blurred = convolve(imgt, kernel, mode='same')

                numnoise = np.where(blurred < 1e-15)

                blurred[numnoise] = 0.0

                for p in range(nbin):

                    sn = sort_list[p]

                    psfout[n, p] = np.sum(blurred[sn])

            self.psfmat = psfout


class SZData:

    def __init__(self, redshift, sz_data=None, rin=None, rout=None, psz=None, covmat_sz=None, cosmo=None):

        if sz_data is None and psz is None:

            print('No data provided. Please provide either an input FITS data file or a set of numpy arrays.')

            return

        if sz_data is not None and psz is not None:

            print('Ambiguous input. Please provide either an input FITS data file or a set of numpy arrays.')

            return

        if sz_data is not None:

            if not os.path.exists(sz_data):

                print('SZ data file not found in path, skipping')

            else:

                print('Reading SZ data file ' + sz_data)

                hdulist = fits.open(sz_data)

                self.pres_sz = hdulist[4].data['FLUX'].reshape(-1)

                self.errp_sz = hdulist[4].data['ERRFLUX'].reshape(-1)

                self.rref_sz = hdulist[4].data['RW'].reshape(-1)

                self.rin = hdulist[4].data['RIN'].reshape(-1)

                self.rout = hdulist[4].data['ROUT'].reshape(-1)

                self.covmat_sz = hdulist[4].data['COVMAT'].reshape(len(self.rref_sz), len(self.rref_sz)).astype(
                    np.float32)

        if psz is not None:

            if rin is None or rout is None or covmat_sz is None:

                print('Missing input, please provide rin, rout, psz, and covmat_sz')

                return

            self.pres_sz = psz

            self.covmat_sz = covmat_sz.astype(np.float32)

            self.errp_sz = np.sqrt(np.diag(covmat_sz))

            if cosmo is None:

                from astropy.cosmology import Planck15 as cosmo

            amin2kpc = cosmo.kpc_proper_per_arcmin(redshift).value

            self.rin_sz = rin * amin2kpc

            self.rout_sz = rout * amin2kpc

            self.rin_sz_am = rin

            self.rout_sz_am = rout

            self.rref_sz = (self.rin_sz + self.rout_sz) / 2.




    def PSF(self, pixsize, psffunc=None, psffile=None, psfimage=None, psfpixsize=None, sourcemodel=None, psfmin=1e-7):
        #####################################################
        # Function to calculate a PSF convolution matrix given an input PSF image or function
        # Images of each annuli are convolved with the PSF image using FFT
        # FFT-convolved images are then used to determine PSF mixing
        #####################################################

        rad = (self.rin_sz_am + self.rout_sz_am) / 2.

        erad = (self.rout_sz_am - self.rin_sz_am) / 2.

        if psffile is None and psfimage is None and psffunc is None:

            print('No PSF model given')

            return

        else:
            if psffile is not None:

                fpsf = fits.open(psffile)

                psfimage = fpsf[0].data.astype(float)

                if psfpixsize is not None:

                    psfpixsize = float(psfimage[0].header['CDELT2'])

                    if psfpixsize == 0.0:

                        print('Error: no pixel size is provided for the PSF image and the CDELT2 keyword is not set')

                        return

                fpsf.close()

            if psfimage is not None:

                if psfpixsize is None or psfpixsize <= 0.0:

                    print('Error: no pixel size is provided for the PSF image')

                    return

            nbin = len(rad)

            psfout = np.zeros((nbin, nbin))

            npexp = 2 * int((rad[nbin - 1] + erad[nbin - 1]) / pixsize) + 1

            exposure = np.ones((npexp, npexp))

            y, x = np.indices(exposure.shape)

            rads = np.hypot(x - npexp / 2., y - npexp / 2.) * pixsize  # arcmin

            kernel = None

            if psffunc is not None:

                kernel = psffunc(rads)

                norm = np.sum(kernel)

                frmax = lambda x: psffunc(x) * 2. * np.pi * x / norm - psfmin

                if frmax(exposure.shape[0] / 2) < 0.:

                    rmax = brentq(frmax, 1., exposure.shape[0]) / pixsize  # pixsize

                    npix = int(rmax)

                else:
                    npix = int(exposure.shape[0] / 2)

                yp, xp = np.indices((2 * npix + 1, 2 * npix + 1))

                rpix = np.sqrt((xp - npix) ** 2 + (yp - npix) ** 2) * pixsize

                kernel = psffunc(rpix)

                norm = np.sum(kernel)

                kernel = kernel / norm

            if psfimage is not None:

                norm = np.sum(psfimage)

                kernel = psfimage / norm

            if kernel is None:

                print('No kernel provided, bye bye')

                return

            # Sort pixels into radial bins
            tol = 0.5e-5

            sort_list = []

            for n in range(nbin):

                if n == 0:

                    sort_list.append(np.where(np.logical_and(rads >= 0, rads < np.round(rad[n] + erad[n], 5) + tol)))

                else:

                    sort_list.append(np.where(np.logical_and(rads >= np.round(rad[n] - erad[n], 5) + tol,
                                                             rads < np.round(rad[n] + erad[n], 5) + tol)))

            # Calculate average of PSF image pixel-by-pixel and sort it by radial bins
            for n in range(nbin):

                # print('Working with bin',n+1)
                region = sort_list[n]

                npt = len(x[region])

                imgt = np.zeros(exposure.shape)

                if sourcemodel is None or sourcemodel.params is None:

                    imgt[region] = 1. / npt

                else:

                    imgt[region] = sourcemodel.model(rads[region], *sourcemodel.params)

                    norm = np.sum(imgt[region])

                    imgt[region] = imgt[region] / norm

                # FFT-convolve image with kernel
                blurred = convolve(imgt, kernel, mode='same')

                numnoise = np.where(blurred < 1e-15)

                blurred[numnoise] = 0.0

                for p in range(nbin):

                    sn = sort_list[p]

                    psfout[n, p] = np.sum(blurred[sn])

            self.psfmat = psfout
