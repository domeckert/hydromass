import corner
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
from scipy.interpolate import InterpolatedUnivariateSpline

from scipy.ndimage import gaussian_filter

def Interp(x_old,y_old,x_new):
    s = InterpolatedUnivariateSpline(x_old, y_old, k=1)
    y_new = s(x_new)
    return y_new

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

    def __init__(self, data=np.random.multivariate_normal([5, 2], [[.5, -0.5], [-0.5, 1.5]],10000), labels=['A','B'],toplot=[0,1],outname='test.pdf',nbins=20,smo=1,coloured=True,title=None, verbose=True,error_on_top=True, pad=10, y=1.05):
        #toplot gives which column to use in plotting
        #labels gives the labels of each column
        #data is the array of data as in example above
        #outname is the name of the file to create
        #nbins sets the number of bins to use for posterior distributions
        #smo indicates how to smooth the coloured data points
        #coloured set indicates whether use colors for data or not
        #title sets the title of the figure to put boldface on top
        #verbose spits out some cute sentences
        #error_on_top puts "variable = value +- error" on top of each plot
        if verbose == True:
            if title is not None:
                print('Producing output corner plot called \"'+outname+'\" and titled: \"'+title+"\"")
            else:
                print('Producing output corner plot called \"'+outname+'\" and without a title')
            print('Using %.0f bins' % nbins)
            print('Using a gaussian smoothing of %.0f' % smo)
            if coloured == True:
                print('Colouring beautifully everything')
            else:
                print('Without coloring a thing :-( sad')
        fig = pl.figure(figsize=(14,14))
        nplots=len(toplot)
        if data.all() == 0:
            data=np.random.randn(10000,2)
            print('creating an example of output in test.pdf')

        fsize=20
        if nplots == 2:
            fsize=32
        if nplots == 3:
            fsize=15
        if nplots == 4:
            fsize=12
        if nplots == 5:
            fsize=10
        if nplots == 6:
            fsize=9
        if nplots >= 7:
            fsize=8

        for ii in range(nplots):
          for jj in range(ii, nplots):
            i=toplot[ii]
            j=toplot[jj]
            if i == j:
                    ax1 = fig.add_subplot(nplots,nplots,1+(nplots+1)*ii)
                    xxx=data[:,i]
                    yy,xxt=np.histogram(xxx, bins=nbins, density=True)
                    xx=(xxt[0:len(yy)]+xxt[1:len(yy)+1])/2.
                    ax1.plot(xx, yy, "-k",color='k',linewidth=2)
                    #ax1.hist(xxx,normed=True,bins=nbins,histtype='step',color='k',linewidth=2)
                    if ii != 0:	ax1.set_yticklabels([])
                    if ii != nplots-1:	ax1.set_xticklabels([])
                    lo, me ,hi = np.percentile(xxx, [16,50,84],axis=0)
                    if error_on_top == True:
                        order_mag=int(np.log10(abs(me)))
                        if order_mag == 0:
                            ax1.set_title(labels[i]+'=%.2f $\pm$ %.2f' % (me,(hi-lo)/2),fontsize=fsize, y=y)
                        else:
                            ax1.set_title(labels[i]+'=(%.2f $\pm$ %.2f)$ \cdot 10^{%d}$' % (me*10**(-order_mag),(hi-lo)/2*10**(-order_mag),order_mag),fontsize=fsize, y=y)

                    xxi=np.linspace(lo,hi,100)
                    yyi=Interp(xx,yy,xxi)
                    mi,ma=np.percentile(xxx, [1,99],axis=0)
                    maxd=Interp(xx,yy,me)
                    #ax1.axvline(me, color='k', linestyle='-')
                    ax1.vlines(x=me, ymin=0, ymax=maxd, color='k', linewidth=2)
                    if coloured == True:
                        ax1.fill_between(xxi,0,yyi,color='cornflowerblue',alpha=0.7,hatch='//')
                    ax1.set_xlim([mi,ma])
                    ax1.set_ylim([0,np.max(yy)*1.1])
                    ax1.tick_params(which='major',length=4,labelsize=22,width=2,direction='in',right=True,top=True,pad=pad)
                    ax1.tick_params(which='minor',length=2,labelsize=10,width=1.5,direction='in',right=True,top=True,pad=pad)
                    ax1.xaxis.set_major_locator(MaxNLocator(2))
                    ax1.yaxis.set_major_locator(MaxNLocator(2))
                    ax1.tick_params(which='major',length=5,labelsize=22,width=2,direction='in',right=True,top=True,pad=pad)
                    ax1.tick_params(which='minor',length=2.5,labelsize=10,width=1.5,direction='in',right=True,top=True,pad=pad)
                    ax1.minorticks_on()
                    if j == toplot[-1]:
                        ax1.set_xlabel(labels[i],fontsize=32)
                    if i == 0:
                        ax1.set_ylabel(labels[i],fontsize=32)


            else:
                xxx=data[:,i]
                yyy=data[:,j]
                if coloured == True:
                    H, X, Y = np.histogram2d(xxx.flatten(), yyy.flatten(), bins=20,  weights=None)
                    ax = fig.add_subplot(nplots,nplots,ii+1+nplots*jj, xlim=X[[0, -1]], ylim=Y[[0, -1]])
                    Hs = gaussian_filter(H, 2)
                    Z=Hs.max() - Hs.T
                    im = mpl.image.NonUniformImage(ax, interpolation='bilinear',cmap=light_jet)
                    xcenters = (X[:-1] + X[1:]) / 2
                    ycenters = (Y[:-1] + Y[1:]) / 2
                    im.set_data(xcenters, ycenters, Z)
                    ax.images.append(im)
                else:
                    ax=fig.add_subplot(nplots,nplots,ii+1+nplots*jj)

                mx=np.median(xxx)
                my=np.median(yyy)
                if jj != nplots-1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel(labels[i],fontsize=32)
                if ii == 0:
                    ax.set_ylabel(labels[j],fontsize=32)
                else:
                    ax.set_yticklabels([])
                corner.hist2d(xxx,yyy,ax=ax,plot_datapoints=False,plot_density=False,levels=(0.393,0.675,0.865),smooth=(smo,smo),bins=(nbins,nbins),no_fill_contours=True)
                if coloured == True:
                    ax.errorbar(mx,my,fmt='D',color='darkorange')
                mi,ma=np.percentile(xxx, [1,99],axis=0)
                ax.set_xlim([mi,ma])
                mi2,ma2=np.percentile(yyy, [1,99],axis=0)
                ax.set_ylim([mi2,ma2])
                ax.xaxis.set_major_locator(MaxNLocator(2))
                ax.yaxis.set_major_locator(MaxNLocator(2))
                ax.tick_params(which='major',length=5,labelsize=22,width=2,direction='in',right=True,top=True,pad=pad)
                ax.tick_params(which='minor',length=2.5,labelsize=10,width=1.5,direction='in',right=True,top=True,pad=pad)
                ax.minorticks_on()
        if title is not None and error_on_top == True:
            top=0.9
        else:
            if error_on_top == True:
                top=0.95
            else:
                if title == '':
                    top=0.99
                else:
                    top=0.95
        pl.subplots_adjust(left=0.11, bottom=0.08, right=0.96, top=top,wspace=0.05, hspace=0.05)
        pl.suptitle(title, fontsize=16,fontweight='bold')

        fig.savefig(outname)








