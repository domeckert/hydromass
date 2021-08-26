import numpy as np
import matplotlib.pyplot as plt
from .deproject import *
from .constants import *
from .pnt import *
from scipy.interpolate import interp1d

def get_coolfunc(Z):
    """
    Retrieve the cooling function data from the internal data archive and return the cooling function profile for a given input metallicity

    :param Z: Metallicity (relative to Anders & Grevesse; defaults to 0.3)
    :type Z: float
    :return: Cooling function profile and temperature grid
    :rtype: numpy.ndarray
    """
    Z_grid = np.array([0., 0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.35, 0.40, 0.48, 0.58, 0.70, 0.85, 1.0])

    file_cf = get_data_file_path('coolfunc_table.fits')

    fcf = fits.open(file_cf)

    ind = np.searchsorted(Z_grid, Z, side='left')

    if Z_grid[ind] < Z:

        Zlow = Z_grid[ind]

        Zhigh = Z_grid[ind + 1]

    elif Z_grid[ind] > Z:

        Zlow = Z_grid[ind - 1]

        Zhigh = Z_grid[ind]

    else:

        Zlow = Z

        Zhigh = Z

    if Zlow != Zhigh:

        hdulow = fcf['COOLFUNC_Z%1.2lf' % (Zlow)]

        lambdalow = hdulow.data['LAMBDA']

        hduhigh = fcf['COOLFUNC_Z%1.2lf' % (Zhigh)]

        lambdahigh = hduhigh.data['LAMBDA']

        lambda_interp = lambdalow + (Z - Zlow) * (lambdahigh - lambdalow) / (Zhigh - Zlow)

    else:

        thdu = fcf['COOLFUNC_Z%1.2lf' % (Z)]

        lambda_interp = thdu.data['LAMBDA']

    return lambda_interp

def cumsum_mat(nval):
    """

    Function to create a matrix that flips a vector, makes a cumulative sum and adds a 0 as the first element
    Then dot(mat, vector) returns the desired vector

    :param nval: (int) Vector size
    :return: (2d-array) Cumulative sum operator
    """
    onemat = np.ones((nval - 1, nval - 1))

    triu = np.triu(onemat)

    flipped_triu = np.flipud(triu)

    zeromat = np.zeros((nval, nval))

    zeromat[1:, 1:] = flipped_triu

    totmat = np.flipud(zeromat)

    return  totmat


def rads_more(Mhyd, nmore=5):
    """

    Return grid of (in, out) radii from X-ray, SZ data or both. Concatenates radii if necessary, then computes a grid of radii.
    Returns the output arrays and the indices corresponding to the input X-ray and/or SZ radii.

    :param Mhyd: (hydromass.Mhyd) Hydromass class containing loaded X-ray and/or SZ loaded data.
    :param nmore: Number of subgrid values to compute the fine binning. Each input bin will be split into nmore values. Default = 20.
    :return: rin, rout, index_x, index_sz, with rin, rout the grids of fine binning, and index_x, index_sz the indices corresponding to the actual input values
    """
    if Mhyd.spec_data is not None and Mhyd.sz_data is None:

        rout_joint = Mhyd.spec_data.rout_x

        rref_joint = Mhyd.spec_data.rref_x

    elif Mhyd.spec_data is None and Mhyd.sz_data is not None:

        rout_joint = Mhyd.sz_data.rout_sz

        rref_joint = Mhyd.sz_data.rref_sz

    elif Mhyd.spec_data is not None and Mhyd.sz_data is not None:

        rout_joint = np.sort(np.append(Mhyd.spec_data.rout_x, Mhyd.sz_data.rout_sz))

        rref_joint = np.sort(np.append(Mhyd.spec_data.rref_x, Mhyd.sz_data.rref_sz))

    else:

        print('No loaded data found in input hydromass.Mhyd object, nothing to do')

        return

    rin_joint = np.roll(rout_joint, 1)

    rin_joint[0] = 0.

    njoint = len(rref_joint)

    tot_joint = np.sort(np.append(rin_joint, rref_joint))

    ntotjoint = len(tot_joint)

    rout_more = np.empty(int((ntotjoint - 0.5) * nmore))

    for i in range(ntotjoint - 1):

        rout_more[i * nmore:(i + 1) * nmore] = np.linspace(tot_joint[i], tot_joint[i + 1], nmore + 1)[1:]

    rout_more[(ntotjoint - 1) * nmore:] = np.linspace(rref_joint[njoint - 1], rout_joint[njoint - 1], int(nmore / 2.) + 1)[1:]

    rin_more = np.roll(rout_more, 1)

    rin_more[0] = 0

    index_x, index_sz = None, None

    if Mhyd.spec_data is not None:

        index_x = np.where(np.in1d(rout_more, Mhyd.spec_data.rref_x))

    if Mhyd.sz_data is not None:

        index_sz = np.where(np.in1d(rout_more, Mhyd.sz_data.rref_sz))

    sum_mat = None

    if Mhyd.spec_data is not None:

        ntot = len(rout_more)

        nx = len(Mhyd.spec_data.rref_x)

        sum_mat = np.zeros((nx, ntot))

        for i in range(nx):

            ix = np.where(np.logical_and(rin_more < Mhyd.spec_data.rout_x[i], rin_more >= Mhyd.spec_data.rin_x[i]))

            nval = len(ix[0])

            sum_mat[i, :][ix] = 1. / nval

    return rin_more, rout_more, index_x, index_sz, sum_mat


def densout_pout_from_samples(Mhyd, model, rin_m, rout_m):

    samples = Mhyd.samples

    nsamp = len(samples)

    nvalm = len(rin_m)

    if Mhyd.cf_prof is not None:

        cf_prof = np.repeat(Mhyd.cf_prof, nsamp).reshape(nvalm, nsamp)

    else:

        cf_prof = Mhyd.ccf

    dens_m = np.sqrt(np.dot(Mhyd.Kdens_m, np.exp(samples.T)) / cf_prof * Mhyd.transf)

    mass = Mhyd.mfact * model.func_np(rout_m, Mhyd.samppar, delta=model.delta) / Mhyd.mfact0

    rout_mul = np.tile(rout_m, nsamp).reshape(nsamp, nvalm)

    rin_mul = np.tile(rin_m, nsamp).reshape(nsamp, nvalm)

    # Adding baryonic mass contribution in case of DM-only fit
    if Mhyd.dmonly:
        # Matrix containing integration volumes
        volmat = np.repeat(4. / 3. * np.pi * (rout_m ** 3 - rin_m ** 3), nsamp).reshape(nvalm, nsamp)

        # Compute Mgas profile as cumulative sum over the volume

        nhconv = cgsamu * Mhyd.mu_e * cgskpc ** 3 / Msun  # Msun/kpc^3

        ones_mat = np.ones((nvalm, nvalm))

        cs_mat = np.tril(ones_mat)

        mgas = np.dot(cs_mat, dens_m * nhconv * volmat) / 1e13 / Mhyd.mfact0

        if Mhyd.mstar is not None:

            r_mstar = Mhyd.mstar[:, 0]

            cum_mstar = Mhyd.mstar[:, 1]

            mstar_m = np.interp(rout_m, r_mstar, cum_mstar)

            mstar_mul = np.repeat(mstar_m, nsamp).reshape(nvalm, nsamp)

            mbar = mgas + mstar_mul / Mhyd.mfact0 / 1e13

        else:

            mbar = mgas

        mass = mass + mbar.T

    # Pressure gradient
    dpres = - mass / rout_mul ** 2 * dens_m.T * (rout_mul - rin_mul)

    press00 = np.exp(Mhyd.samplogp0)

    int_mat = cumsum_mat(nvalm)

    press_out = press00 - np.dot(int_mat, dpres.T)

    if Mhyd.pnt:

        alpha_turb = alpha_turb_np(rout_m, Mhyd.samppar, Mhyd.redshift, Mhyd.pnt_pars)

        pth = press_out * (1. - alpha_turb)

    else:

        pth = press_out

    return  dens_m, press_out, pth


def kt_from_samples(Mhyd, model, nmore=5):
    """

    Compute model temperature profile from Mhyd reconstruction evaluated at reference X-ray temperature radii

    :param Mhyd: mhyd.Mhyd object including the reconstruction
    :param model: mhyd.Model object defining the mass model
    :return: Median temperature, Lower 1-sigma percentile, Upper 1-sigma percentile
    """

    if Mhyd.spec_data is None:

        print('No spectral data provided')

        return

    nsamp = len(Mhyd.samples)

    rin_m, rout_m, index_x, index_sz, sum_mat = rads_more(Mhyd, nmore=nmore)

    vx = MyDeprojVol(rin_m / Mhyd.amin2kpc, rout_m / Mhyd.amin2kpc)

    vol_x = vx.deproj_vol().T

    if Mhyd.spec_data.psfmat is not None:

        mat1 = np.dot(Mhyd.spec_data.psfmat.T, sum_mat)

        proj_mat = np.dot(mat1, vol_x)

    else:

        proj_mat = np.dot(sum_mat, vol_x)

    npx = len(Mhyd.spec_data.rref_x)

    dens_m, press_out, pth = densout_pout_from_samples(Mhyd, model, rin_m, rout_m)

    t3d = pth / dens_m

    # Mazzotta weights
    ei = dens_m ** 2 * t3d ** (-0.75)

    # Temperature projection
    flux = np.dot(proj_mat, ei)

    tproj = np.dot(proj_mat, t3d * ei) / flux

    tmed, tlo, thi = np.percentile(tproj, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    t3dot, t3dlt, t3dht = np.percentile(t3d, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    t3do, t3dl, t3dh = t3dot[index_x], t3dlt[index_x], t3dht[index_x]

    dict = {
        "R_IN": Mhyd.spec_data.rin_x,
        "R_OUT": Mhyd.spec_data.rout_x,
        "R_REF": Mhyd.spec_data.rref_x,
        "T3D": t3do,
        "T3D_LO": t3dl,
        "T3D_HI": t3dh,
        "TSPEC": tmed,
        "TSPEC_LO": tlo,
        "TSPEC_HI": thi
    }

    return dict


def P_from_samples(Mhyd, model, nmore=5):
    """

    Compute model pressure profile from Mhyd reconstruction evaluated at the reference SZ radii

    :param Mhyd: mhyd.Mhyd object including the reconstruction
    :param model: mhyd.Model object defining the mass model
    :return: Median pressure, Lower 1-sigma percentile, Upper 1-sigma percentile
    """

    if Mhyd.sz_data is None:

        print('No SZ data provided')

        return

    rin_m, rout_m, index_x, index_sz, sum_mat = rads_more(Mhyd, nmore=nmore)

    dens_m, press_tot, pth = densout_pout_from_samples(Mhyd, model, rin_m, rout_m)

    pmt, plot, phit = np.percentile(pth, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    pmed, plo, phi = pmt[index_sz], plot[index_sz], phit[index_sz]

    return pmed, plo, phi


def mass_from_samples(Mhyd, model, nmore=5, plot=False):
    """

    Compute median and percentile mass profile, gas mass and gas fraction from Mhyd reconstruction

    :param Mhyd: mhyd.Mhyd object including the reconstruction
    :param model: mhyd.Model object defining the mass model
    :return: Median mass [in M_sun], Lower 1-sigma percentile, Upper 1-sigma percentile, Median Mgas, Lower, Upper, Median Fgas, Lower, Upper
    """

    rin_m, rout_m, index_x, index_sz, sum_mat = rads_more(Mhyd, nmore=nmore)

    mass = Mhyd.mfact * model.func_np(rout_m, Mhyd.samppar, model.delta) * 1e13

    nsamp = len(Mhyd.samppar)

    nvalm = len(rin_m)

    if Mhyd.cf_prof is not None:

        cf_prof = np.repeat(Mhyd.cf_prof, nsamp).reshape(nvalm, nsamp)

    else:

        cf_prof = Mhyd.ccf

    alldens = np.sqrt(np.dot(Mhyd.Kdens_m, np.exp(Mhyd.samples.T)) / cf_prof * Mhyd.transf)

    # Matrix containing integration volumes
    volmat = np.repeat(4. / 3. * np.pi * (rout_m ** 3 - rin_m ** 3), nsamp).reshape(nvalm, nsamp)

    # Compute Mgas profile as cumulative sum over the volume

    nhconv = cgsamu * Mhyd.mu_e * cgskpc ** 3 / Msun  # Msun/kpc^3

    ones_mat = np.ones((nvalm, nvalm))

    cs_mat = np.tril(ones_mat)

    mgas = np.dot(cs_mat, alldens * nhconv * volmat)

    mg, mgl, mgh = np.percentile(mgas, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    if Mhyd.mstar is not None:

        r_mstar = Mhyd.mstar[:, 0]

        cum_mstar = Mhyd.mstar[:, 1]

        mstar_m = np.interp(rout_m, r_mstar, cum_mstar)

    else:

        mstar_m = np.zeros(nvalm)

    if Mhyd.dmonly:

        mtot = mass + mgas.T + mstar_m.T

        fgas = mgas / mtot.T

    else:

        fgas = mgas / mass.T

    fg, fgl, fgh = np.percentile(fgas, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    mmed, mlo, mhi = np.percentile(mass, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=0)

    if Mhyd.dmonly:

        mtotm, mtotlo, mtothi = np.percentile(mtot, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=0)

    else:

        mtotm, mtotlo, mtothi = mmed, mlo, mhi

    dict = {
        "R_IN": rin_m,
        "R_OUT": rout_m,
        "MASS": mtotm,
        "MASS_LO": mtotlo,
        "MASS_HI": mtothi,
        "M_DM": mmed,
        "M_DM_LO": mlo,
        "M_DM_HI": mhi,
        "MGAS": mg,
        "MGAS_LO": mgl,
        "MGAS_HI": mgh,
        "FGAS": fg,
        "FGAS_LO": fgl,
        "FGAS_HI": fgh,
        "M_STAR": mstar_m
    }

    if plot:

        fig = plt.figure(figsize=(13, 10))

        ax_size = [0.14, 0.12,
                   0.85, 0.85]

        ax = fig.add_axes(ax_size)

        ax.minorticks_on()

        ax.tick_params(length=20, width=1, which='major', direction='in', right=True, top=True)

        ax.tick_params(length=10, width=1, which='minor', direction='in', right=True, top=True)

        for item in (ax.get_xticklabels() + ax.get_yticklabels()):

            item.set_fontsize(22)

        plt.xscale('log')

        plt.yscale('log')

        plt.plot(rout_m, mg, color='blue', label='$M_{gas}$')

        plt.fill_between(rout_m, mgl, mgh, color='blue', alpha=0.4)

        plt.plot(rout_m, mmed, color='red', label='$M_{Hyd}$')

        plt.fill_between(rout_m, mlo, mhi, color='red', alpha=0.4)

        if Mhyd.mstar is not None:

            plt.plot(rout_m, mstar_m, color='green', label='$M_{\star}$')

        plt.xlabel('Radius [kpc]', fontsize=40)

        plt.ylabel('$M(<R) [M_\odot]$', fontsize=40)

        plt.legend(fontsize = 22)

        return dict, fig

    else:

        return dict



def prof_hires(Mhyd, model, nmore=5, Z=0.3):
    """
    Compute best-fitting profiles and error envelopes from fitted data

    :param Mhyd: (hydromass.Mhyd) Object containing results of mass reconstruction
    :param model:
    :param nmore:
    :return:
    """

    rin_m, rout_m, index_x, index_sz, sum_mat = rads_more(Mhyd, nmore=nmore)

    vx = MyDeprojVol(rin_m / Mhyd.amin2kpc, rout_m / Mhyd.amin2kpc)

    vol_x = vx.deproj_vol().T

    dens_m, p3d, pth = densout_pout_from_samples(Mhyd, model, rin_m, rout_m)

    t3d = pth / dens_m

    # Mazzotta weights
    ei = dens_m ** 2 * t3d ** (-0.75)

    # Temperature projection
    flux = np.dot(vol_x, ei)

    tproj = np.dot(vol_x, t3d * ei) / flux

    K3d = t3d * dens_m ** (- 2. / 3.)

    mptot, mptotl, mptoth = np.percentile(p3d, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    mp, mpl, mph = np.percentile(pth, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    mt3d, mt3dl, mt3dh = np.percentile(t3d, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    mtp, mtpl, mtph = np.percentile(tproj, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    mne, mnel, mneh = np.percentile(dens_m, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    mK, mKl, mKh = np.percentile(K3d, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    coolfunc, ktgrid = get_coolfunc(Z)

    lambda3d = np.interp(t3d, ktgrid, coolfunc)

    tcool = 3./2. * dens_m * (1. + 1./Mhyd.nhc) * t3d * kev2erg / (lambda3d * dens_m **2 / Mhyd.nhc)

    mtc, mtcl, mtch = np.percentile(tcool, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    mcf, mcfl, mcfh = np.percentile(lambda3d, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    if Mhyd.pnt:

        pnt_all = p3d - pth

        mpnt, mpntl, mpnth = np.percentile(pnt_all, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    else:

        mpnt, mpntl, mpnth = np.zeros(len(mptot)), np.zeros(len(mptot)), np.zeros(len(mptot))

    dict={
        "R_IN": rin_m,
        "R_OUT": rout_m,
        "P_TOT": mptot,
        "P_TOT_LO": mptotl,
        "P_TOT_HI": mptoth,
        "P_TH": mp,
        "P_TH_LO": mpl,
        "P_TH_HI": mph,
        "T3D": mt3d,
        "T3D_LO": mt3dl,
        "T3D_HI": mt3dh,
        "TSPEC": mtp,
        "TSPEC_LO": mtpl,
        "TSPEC_HI": mtph,
        "NE": mne,
        "NE_LO": mnel,
        "NE_HI": mneh,
        "K": mK,
        "K_LO": mKl,
        "K_HI": mKh,
        "P_NT": mpnt,
        "P_NT_LO": mpntl,
        "P_NT_HI": mpnth,
        "T_COOL": mtc,
        "T_COOL_LO": mtcl,
        "T_COOL_HI": mtch,
        "LAMBDA": mcf,
        "LAMBDA_LO": mcfl,
        "LAMBDA_HI": mcfh
    }

    return dict


def mgas_pm(rin_m, rout_m, dens):

    # Integration volumes
    volint = 4. /3. * np.pi * (rout_m ** 3 - rin_m ** 3)

    nvalm = len(rin_m)

    ones_mat = np.ones((nvalm, nvalm))

    cs_mat = np.tril(ones_mat)

    mgas = pm.math.dot(cs_mat, dens * volint) / 1e13

    return mgas



def PlotMgas(Mhyd, plot=False, outfile=None, nmore=5):
    """
    Compute Mgas profile and error envelope from mass reconstruction run

    :param Mhyd: (hydromass.Mhyd) Mhyd object containing the results of the mass reconstruction
    :param plot: (bool) Plot the gas mass profile (default=False)
    :param outfile: (str) If plot=True, file name to output the plotted Mgas profile (default=None)
    :param nmore: (int) Number of points defining fine grid, must be equal to the value used for the mass reconstruction (default=5)
    :return:
    """

    if Mhyd.samples is None or Mhyd.redshift is None or Mhyd.ccf is None:

        print('Error: no mass reconstruction found')

        return

    nsamp = len(Mhyd.samples)

    rin_m, rout_m, index_x, index_sz, sum_mat = rads_more(Mhyd, nmore=nmore)

    nvalm = len(rin_m)

    if Mhyd.cf_prof is not None:

        cf_prof = np.repeat(Mhyd.cf_prof, nsamp).reshape(nvalm, nsamp)

    else:

        cf_prof = Mhyd.ccf

    alldens = np.sqrt(np.dot(Mhyd.Kdens_m, np.exp(Mhyd.samples.T)) / cf_prof * Mhyd.transf)

    # Matrix containing integration volumes
    volmat = np.repeat(4. / 3. * np.pi * (rout_m ** 3 - rin_m ** 3), nsamp).reshape(nvalm, nsamp)

    # Compute Mgas profile as cumulative sum over the volume

    nhconv = cgsamu * Mhyd.mu_e * cgskpc ** 3 / Msun  # Msun/kpc^3

    ones_mat = np.ones((nvalm, nvalm))

    cs_mat = np.tril(ones_mat)

    mgas = np.dot(cs_mat, alldens * nhconv * volmat)

    mg, mgl, mgh = np.percentile(mgas, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    if plot:

        fig = plt.figure(figsize=(13, 10))

        ax_size = [0.14, 0.12,
                   0.85, 0.85]

        ax = fig.add_axes(ax_size)

        ax.minorticks_on()

        ax.tick_params(length=20, width=1, which='major', direction='in', right=True, top=True)

        ax.tick_params(length=10, width=1, which='minor', direction='in', right=True, top=True)

        for item in (ax.get_xticklabels() + ax.get_yticklabels()):

            item.set_fontsize(22)

        plt.xscale('log')

        plt.yscale('log')

        plt.plot(rout_m, mg, color='blue')

        plt.fill_between(rout_m, mgl, mgh, color='blue', alpha=0.4)

        plt.xlabel('Radius [kpc]', fontsize=40)

        plt.ylabel('$M_{gas} [M_\odot]$', fontsize=40)

        if outfile is not None:
            plt.savefig(outfile)
            plt.close()

        return mg, mgl, mgh, fig

    else:

        return mg, mgl, mgh