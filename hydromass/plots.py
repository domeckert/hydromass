import numpy as np
import matplotlib.pyplot as plt
from .deproject import *
from .constants import *
from scipy.interpolate import interp1d

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
    if Mhyd.rout_x is not None and Mhyd.rout_sz is None:

        rout_joint = Mhyd.rout_x

        rref_joint = Mhyd.rref_x

    elif Mhyd.rout_x is None and Mhyd.rout_sz is not None:

        rout_joint = Mhyd.rout_sz

        rref_joint = Mhyd.rref_sz

    elif Mhyd.rout_x is not None and Mhyd.rout_sz is not None:

        rout_joint = np.sort(np.append(Mhyd.rout_x, Mhyd.rout_sz))

        rref_joint = np.sort(np.append(Mhyd.rref_x, Mhyd.rref_sz))

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

    if Mhyd.rout_x is not None:

        index_x = np.where(np.in1d(rin_more, Mhyd.rref_x))

    if Mhyd.rout_sz is not None:

        index_sz = np.where(np.in1d(rin_more, Mhyd.rref_sz))

    return rin_more, rout_more, index_x, index_sz


def densout_pout_from_samples(Mhyd, model, rin_m, rout_m):

    samples = Mhyd.samples

    nsamp = len(samples)

    nvalm = len(rin_m)

    dens_m = np.sqrt(np.dot(Mhyd.Kdens_m, np.exp(samples.T)) / Mhyd.ccf * Mhyd.transf)

    mass = Mhyd.mfact * model.func_np(rout_m, Mhyd.samppar, delta=model.delta) / Mhyd.mfact0

    rout_mul = np.tile(rout_m, nsamp).reshape(nsamp, nvalm)

    rin_mul = np.tile(rin_m, nsamp).reshape(nsamp, nvalm)

    # Pressure gradient
    dpres = - mass / rout_mul ** 2 * dens_m.T * (rout_mul - rin_mul)

    press00 = np.exp(Mhyd.samplogp0)

    int_mat = cumsum_mat(nvalm)

    press_out = press00 - np.dot(int_mat, dpres.T)

    return  dens_m, press_out


def kt_from_samples(Mhyd, model, nmore=5):
    """

    Compute model temperature profile from Mhyd reconstruction evaluated at reference X-ray temperature radii

    :param Mhyd: mhyd.Mhyd object including the reconstruction
    :param model: mhyd.Model object defining the mass model
    :return: Median temperature, Lower 1-sigma percentile, Upper 1-sigma percentile
    """

    if Mhyd.temp_x is None:

        print('No spectral data provided')

        return

    nsamp = len(Mhyd.samples)

    rin_m, rout_m, index_x, index_sz = rads_more(Mhyd, nmore=nmore)

    vx = MyDeprojVol(rin_m / Mhyd.amin2kpc, rout_m / Mhyd.amin2kpc)

    vol_x = vx.deproj_vol().T

    npx = len(Mhyd.rref_x)

    dens_m, press_out = densout_pout_from_samples(Mhyd, model, rin_m, rout_m)

    t3d = press_out / dens_m

    # Mazzotta weights
    ei = dens_m ** 2 * t3d ** (-0.75)

    # Temperature projection
    flux = np.dot(vol_x, ei)

    tproj = np.dot(vol_x, t3d * ei) / flux

    tfit_proj, t3d_or = np.empty((npx, nsamp)), np.empty((npx, nsamp))

    for i in range(nsamp):

        tfit_proj[:, i] = tproj[:, i][index_x]

        t3d_or[:, i] = t3d[:, i][index_x]

    tmed, tlo, thi = np.percentile(tfit_proj, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    t3do, t3dl, t3dh = np.percentile(t3d_or,[50., 50. - 68.3 / 2. , 50. + 68.3 / 2.] , axis=1)

    dict = {
        "R_IN": Mhyd.rin_x,
        "R_OUT": Mhyd.rout_x,
        "R_REF": Mhyd.rref_x,
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

    if Mhyd.pres_sz is None:

        print('No SZ data provided')

        return

    nsamp = len(Mhyd.samples)

    rin_m, rout_m, index_x, index_sz = rads_more(Mhyd, nmore=nmore)

    npx = len(Mhyd.rref_sz)

    dens_m, press_sz = densout_pout_from_samples(Mhyd, model, rin_m, rout_m)

    pfit = np.empty((npx, nsamp))

    for i in range(nsamp):

        pfit[:, i] = press_sz[:, i][index_sz]

    pmed, plo, phi = np.percentile(pfit, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    return pmed, plo, phi


def mass_from_samples(Mhyd, model, nmore=5, plot=False):
    """

    Compute median and percentile mass profile, gas mass and gas fraction from Mhyd reconstruction

    :param Mhyd: mhyd.Mhyd object including the reconstruction
    :param model: mhyd.Model object defining the mass model
    :return: Median mass [in M_sun], Lower 1-sigma percentile, Upper 1-sigma percentile, Median Mgas, Lower, Upper, Median Fgas, Lower, Upper
    """

    rin_m, rout_m, index_x, index_sz = rads_more(Mhyd, nmore=nmore)

    mass = Mhyd.mfact * model.func_np(rout_m, Mhyd.samppar, model.delta) * 1e13

    mmed, mlo, mhi = np.percentile(mass, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=0)

    nsamp = len(Mhyd.samppar)

    nvalm = len(rin_m)

    alldens = np.sqrt(np.dot(Mhyd.Kdens_m, np.exp(Mhyd.samples.T)) / Mhyd.ccf * Mhyd.transf)

    # Matrix containing integration volumes
    volmat = np.repeat(4. / 3. * np.pi * (rout_m ** 3 - rin_m ** 3), nsamp).reshape(nvalm, nsamp)

    # Compute Mgas profile as cumulative sum over the volume

    nhconv = cgsamu * Mhyd.mu_e * cgskpc ** 3 / Msun  # Msun/kpc^3

    ones_mat = np.ones((nvalm, nvalm))

    cs_mat = np.tril(ones_mat)

    mgas = np.dot(cs_mat, alldens * nhconv * volmat)

    mg, mgl, mgh = np.percentile(mgas, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    fgas = mgas / mass.T

    fg, fgl, fgh = np.percentile(fgas, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    dict = {
        "R_IN": rin_m,
        "R_OUT": rout_m,
        "MASS": mmed,
        "MASS_LO": mlo,
        "MASS_HI": mhi,
        "MGAS": mg,
        "MGAS_LO": mgl,
        "MGAS_HI": mgh,
        "FGAS": fg,
        "FGAS_LO": fgl,
        "FGAS_HI": fgh
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

        plt.plot(rout_m, mg, color='blue', label='$M_{\rm gas}$')

        plt.fill_between(rout_m, mgl, mgh, color='blue', alpha=0.4)

        plt.plot(rout_m, mmed, color='red', label='$M_{\rm Hyd}$')

        plt.fill_between(rout_m, mlo, mhi, color='red', alpha=0.4)

        plt.xlabel('Radius [kpc]', fontsize=40)

        plt.ylabel('$M(<R) [M_\odot]$', fontsize=40)

        return dict, fig

    else:

        return dict



def prof_hires(Mhyd, model, nmore=5):
    """
    Compute best-fitting profiles and error envelopes from fitted data

    :param Mhyd: (hydromass.Mhyd) Object containing results of mass reconstruction
    :param model:
    :param nmore:
    :return:
    """

    rin_m, rout_m, index_x, index_sz = rads_more(Mhyd, nmore=nmore)

    nhires = len(rin_m)

    vx = MyDeprojVol(rin_m / Mhyd.amin2kpc, rout_m / Mhyd.amin2kpc)

    vol_x = vx.deproj_vol().T

    dens_m, p3d = densout_pout_from_samples(Mhyd, model, rin_m, rout_m)

    t3d = p3d / dens_m

    # Mazzotta weights
    ei = dens_m ** 2 * t3d ** (-0.75)

    # Temperature projection
    flux = np.dot(vol_x, ei)

    tproj = np.dot(vol_x, t3d * ei) / flux

    K3d = t3d * dens_m ** (- 2. / 3.)

    mp, mpl, mph = np.percentile(p3d, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    mt3d, mt3dl, mt3dh = np.percentile(t3d, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    mtp, mtpl, mtph = np.percentile(tproj, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    mne, mnel, mneh = np.percentile(dens_m, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    mK, mKl, mKh = np.percentile(K3d, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis=1)

    dict={
        "R_IN": rin_m,
        "R_OUT": rout_m,
        "P": mp,
        "P_LO": mpl,
        "P_HI": mph,
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
        "K_HI": mKh
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

    rin_m, rout_m, index_x, index_sz = rads_more(Mhyd, nmore=nmore)

    nvalm = len(rin_m)

    alldens = np.sqrt(np.dot(Mhyd.Kdens_m, np.exp(Mhyd.samples.T)) / Mhyd.ccf * Mhyd.transf)

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