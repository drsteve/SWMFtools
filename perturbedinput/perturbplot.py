import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import spacepy.plot as splot

space = ' '

def roundUpTo(inval, onein=10):
    '''Round input value UP to nearest (1/onein).
    
    E.g., to get 0.0795 to nearest 0.01 (above), use roundUpTo(0.0795, 100)
    to get 0.51 to nearest 0.1, use roundUpTo(0.51, 10)'''
    return np.ceil(inval*onein)/onein

def plotMesh(xx, yy, zz, cmap='plasma', target=None, horiz=False):
    fig, ax = splot.set_target(target)
    orient = 'horizontal' if horiz else 'vertical'
    map_ax = ax.pcolormesh(xx, yy, zz, cmap=cmap)
    cax = fig.colorbar(map_ax, ax=ax, orientation=orient)
    m0 = 0                      # colorbar min value
    ntens = 10 if (1/cax.vmax <= 10) else 100
    m5 = roundUpTo(cax.vmax, ntens) # colorbar max value
    num_ticks = 5
    newticks = np.linspace(m0, m5, num_ticks)
    cax.set_ticks(newticks.tolist())
    cax.set_ticklabels(['{:.2f}'.format(x) for x in newticks.tolist()])
    return ax, cax

def errorHist2D(errors, depend, plotinfo, cmap='plasma'):
    var = plotinfo['var']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    map_ax = ax.hexbin(errors, depend, mincnt=1, cmap=cmap)
    ax.set_xlim(plotinfo['xlimdict'][var])
    ax.set_xlabel(r'$\varepsilon$' + space + plotinfo['units'][var])
    ax.set_ylim(plotinfo['ylimdict'][var])
    ax.set_ylabel(plotinfo['varlabel'] + space + plotinfo['units'][var])
    cax = fig.colorbar(map_ax)
    caxlabel = 'Count'
    cax.set_label(caxlabel)
    if 'altvar' in plotinfo:
        plt.savefig('Err_hexbin_{}_{}_{}.png'.format(plotinfo['err'], var, plotinfo['altvar']))
    else:
        plt.savefig('Err_hexbin_{}_{}.png'.format(plotinfo['err'], var))
    plt.close(fig)

def errorJointKDE2D(xx, yy, zz, plotinfo, cmap='plasma'):
    var = plotinfo['var']
    ax, cax = plotMesh(xx, yy, zz, cmap=cmap)
    fig = ax.figure
    ax.set_xlim(plotinfo['xlimdict'][var])
    ax.set_xlabel(r'$\varepsilon$' + space + plotinfo['units'][var])
    ax.set_ylim(plotinfo['ylimdict'][var])
    ax.set_ylabel(plotinfo['varlabel'] + space + plotinfo['units'][var])
    caxlabel = r'P($\varepsilon$, '+ plotinfo['varlabel'] +')'
    cax.set_label(caxlabel)

    if 'altvar' in plotinfo:
        plt.savefig('Err_joint_distrib_{}_{}_{}.png'.format(plotinfo['err'], var, plotinfo['altvar']))
    else:
        plt.savefig('Err_joint_distrib_{}_{}.png'.format(plotinfo['err'], var))
    plt.close(fig)

def errorCondKDE2D(xx, yy, zz, plotinfo, cmap='plasma', target=None, save=True):
    var = plotinfo['var']
    hflag = True if (target is not None) else False
    ax, cax = plotMesh(xx, yy, zz, cmap=cmap, target=target, horiz=hflag)
    fig = ax.figure
    if 'annot' in plotinfo:
        ax.text(0.1, 0.9, plotinfo['annot'], fontdict={'color': 'w', 'size': 15}, transform=ax.transAxes)
    ax.set_xlim(plotinfo['xlimdict'][var])
    ax.set_xlabel(r'$\varepsilon$' + space + plotinfo['units'][var])
    ax.set_ylim(plotinfo['ylimdict'][var])
    ax.set_ylabel(plotinfo['varlabel'] + space + plotinfo['units'][var])
    cax.set_label(r'p($\varepsilon$|'+ plotinfo['varlabel'] +')')
    if save:
        if 'altvar' in plotinfo:
            plt.savefig('Err_conditional_{}_{}_{}.png'.format(plotinfo['err'], var, plotinfo['altvar']))
        else:
            plt.savefig('Err_conditional_{}_{}.png'.format(plotinfo['err'], var))
        plt.close(fig)
    else:
        return fig, ax, cax

def plotMarginal(x, marginal, plotinfo):
    var = plotinfo['var']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, marginal)
    ax.set_xlim(plotinfo['ylimdict'][var])
    ax.set_xlabel(plotinfo['varlabel'] + space + plotinfo['units'][var])
    ymax = ax.get_ylim()[1]
    ax.set_ylim([0, ymax])
    ax.set_ylabel('Probability Density')
    plt.title('Marginal Probability, P('+ plotinfo['varlabel'] +')')
    plt.tight_layout()
    if 'altvar' in plotinfo:
        plt.savefig('Err_marginal_{}_{}_{}.png'.format(plotinfo['err'], var, plotinfo['altvar']))
    else:
        plt.savefig('Err_marginal_{}_{}.png'.format(plotinfo['err'], var))
    plt.close(fig)

def plotSelectionTimestep(x, y, val, plotinfo, run_num, timestep):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.vlines(val, 0, 1, transform=ax.get_xaxis_transform(), colors='r')
    ax.set_ylim([0, plotinfo['plim']])
    ax.set_xlim(plotinfo['xlimdict'][plotinfo['var']])
    ax.set_xlabel(plotinfo['errlabel']+plotinfo['varlabel'])
    plt.savefig('{}_step{:02d}.png'.format(plotinfo['var'], timestep))
    plt.close(fig)
