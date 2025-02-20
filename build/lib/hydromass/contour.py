import corner
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
from scipy.interpolate import InterpolatedUnivariateSpline

from scipy.ndimage import gaussian_filter

def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return mpl.colors.LinearSegmentedColormap('colormap',cdict,1024)

def Interp(x_old,y_old,x_new):
    s = InterpolatedUnivariateSpline(x_old, y_old, k=1)
    y_new = s(x_new)
    return y_new

light_jet = cmap_map(lambda x: x*0.75 + 0.25, mpl.cm.Blues_r)


class Contour:
    '''
    Produce corner plots from output samples to show the posterior distributions in the diagonal elements and the 2D contour plots in the non-diagonal elements.

    :param data: Numpy array containing the samples of the posterior distributions for the parameters of interest
    :type data: :class:`numpy.ndarray`
    :param labels: List of parameter names
    :type labels: list
    :param toplot: List containing the columns of the input array that will be plotted
    :type toplot: list
    :param outname: Name of the output file
    :type outname: str
    :param nbins: Number of bins to extract the posterior distributions
    :type nbins: int
    :param smo: Gaussian smoothing kernel size
    :type smo: float
    :param coloured: Set whether the plot will be in color or grayscale
    :type coloured: bool
    :param title: Plot title
    :type title:  str
    :param verbose: Verbose output
    :type verbose: bool
    :param error_on_top: Add a line at the top of each diagonal plot indicating the mean and 1-sigma error on the corresponding parameter
    :type error_on_top: bool
    :param pad: Pad size
    :type pad: int
    :param y: Size of the space between the plot and the error on top (if error_on_top=True)
    :type y: float
    '''

    def __init__(self, data=np.random.multivariate_normal([5, 2], [[.5, -0.5], [-0.5, 1.5]], 10000), labels=['A','B'], toplot=[0,1], outname='test.pdf', nbins=20, smo=1, coloured=True, title=None, verbose=True, error_on_top=True, pad=10, y=1.05):
        if verbose:
            if title:
                print(f'Producing output corner plot called "{outname}" and titled: "{title}"')
            else:
                print(f'Producing output corner plot called "{outname}" without a title')
            print(f'Using {nbins} bins')
            print(f'Using a Gaussian smoothing of {smo}')
            print('Colouring beautifully everything' if coloured else 'Without coloring a thing :-( sad')

        fig = pl.figure(figsize=(14, 14))
        nplots = len(toplot)
        fsize = max(8, 32 - 3 * (nplots - 2))

        if data.all() == 0:
            data = np.random.randn(10000, 2)
            print('Creating an example of output in test.pdf')

        for ii in range(nplots):
            for jj in range(ii, nplots):
                i = toplot[ii]
                j = toplot[jj]

                if i == j:  # Diagonal plot
                    ax1 = fig.add_subplot(nplots, nplots, 1 + (nplots + 1) * ii)
                    xxx = data[:, i]
                    yy, xxt = np.histogram(xxx, bins=nbins, density=True)
                    xx = (xxt[:-1] + xxt[1:]) / 2.
                    ax1.plot(xx, yy, color='k', linewidth=2)

                    if ii != 0:
                        ax1.set_yticklabels([])
                    if ii != nplots - 1:
                        ax1.set_xticklabels([])

                    lo, me, hi = np.percentile(xxx, [16, 50, 84], axis=0)
                    if error_on_top:
                        order_mag = int(np.log10(abs(me)))
                        if order_mag == 0:
                            ax1.set_title(f'{labels[i]}={me:.2f} ± {(hi-lo)/2:.2f}', fontsize=fsize, y=y)
                        else:
                            ax1.set_title(f'{labels[i]}=({me*10**(-order_mag):.2f} ± {(hi-lo)/2*10**(-order_mag):.2f}) × 10^{order_mag}', fontsize=fsize, y=y)

                    xxi = np.linspace(lo, hi, 100)
                    yyi = Interp(xx, yy, xxi)
                    mi, ma = np.percentile(xxx, [1, 99], axis=0)
                    maxd = Interp(xx, yy, me)

                    ax1.vlines(x=me, ymin=0, ymax=maxd, color='k', linewidth=2)
                    if coloured:
                        ax1.fill_between(xxi, 0, yyi, color='cornflowerblue', alpha=0.7, hatch='//')        

                    ax1.set_xlim([mi, ma])
                    ax1.set_ylim([0, max(yy) * 1.1])
                    ax1.xaxis.set_major_locator(MaxNLocator(2))
                    ax1.yaxis.set_major_locator(MaxNLocator(2))
                    ax1.tick_params(which='both', length=5, labelsize=22, width=2, direction='in', pad=pad)
                    ax1.minorticks_on()

                else:  # Off-diagonal plot
                    xxx = data[:, i]
                    yyy = data[:, j]
                    if coloured:
                        H, X, Y = np.histogram2d(xxx.flatten(), yyy.flatten(), bins=20)
                        ax = fig.add_subplot(nplots, nplots, ii + 1 + nplots * jj, xlim=X[[0, -1]], ylim=Y[[0, -1]])
                        Hs = gaussian_filter(H, 2)
                        Z = Hs.max() - Hs.T
                        im = mpl.image.NonUniformImage(ax, interpolation='bilinear', cmap=light_jet)
                        xcenters = (X[:-1] + X[1:]) / 2
                        ycenters = (Y[:-1] + Y[1:]) / 2
                        im.set_data(xcenters, ycenters, Z)
                        ax.add_artist(im)
                    else:
                        ax = fig.add_subplot(nplots, nplots, ii + 1 + nplots * jj)

                    mx = np.median(xxx)
                    my = np.median(yyy)

                    # Set labels only on leftmost edge and bottom row
                    if jj != nplots - 1:
                        ax.set_xticklabels([])
                    else:
                        ax.set_xlabel(labels[i], fontsize=32)

                    if ii == 0:
                        ax.set_ylabel(labels[j], fontsize=32)
                    else:
                        ax.set_yticklabels([])

                    corner.hist2d(xxx, yyy, ax=ax, plot_datapoints=False, plot_density=False, levels=(0.393, 0.675, 0.865), smooth=(smo, smo), bins=(nbins, nbins), no_fill_contours=True)
                    if coloured:
                        ax.errorbar(mx, my, fmt='D', color='darkorange')

                    mi, ma = np.percentile(xxx, [1, 99], axis=0)
                    ax.set_xlim([mi, ma])
                    mi2, ma2 = np.percentile(yyy, [1, 99], axis=0)
                    ax.set_ylim([mi2, ma2])
                    ax.xaxis.set_major_locator(MaxNLocator(2))
                    ax.yaxis.set_major_locator(MaxNLocator(2))
                    ax.tick_params(which='both', length=5, labelsize=22, width=2, direction='in', pad=pad)
                    ax.minorticks_on()
                    if ii == 0:
                        ax.set_ylabel(labels[j], fontsize=32)
                    else:
                        ax.set_yticklabels([])

        
        # Adjust layout and save figure
        top_margin = 0.9 if title and error_on_top else 0.95 if error_on_top else 0.99
        pl.subplots_adjust(left=0.11, bottom=0.08, right=0.96, top=top_margin, wspace=0.05, hspace=0.05)
        pl.suptitle(title, fontsize=16, fontweight='bold')
        fig.savefig(outname)








