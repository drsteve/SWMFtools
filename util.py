import glob
import os
import matplotlib.pyplot as plt
from spacepy.pybats import bats


def load_logs(workdir='.', logtype='log', logbase='log*log', geobase='geo*log'):
    """Load and merge all logfiles associated with a run

    Will merge multiple logs, e.g., from a sequence of restarts
    If logs overlap, e.g., because a run timed out and was restarted from a 
    time before the end of the previous log, then the overlap period is removed
    from the earlier log file.
    """
    workdir = os.path.abspath(workdir)
    if logtype == 'log':
        globbase = os.path.join(workdir, logbase)
        loader = bats.BatsLog
    elif logtype == 'geo':
        globbase = os.path.join(workdir, geobase)
        loader = bats.GeoIndexFile
    else:
        raise ValueError('load_logs: logtype must be either "log" or "geo", not {}'.format(logtype))
    fns = sorted(glob.glob(globbase))
    if not fns:
        raise IOError('No log files found with selected search term ({})'.format(globbase))
    all_logs = [loader(fn) for fn in fns]
    log = all_logs[0]
    if len(all_logs)>1:
        for nlg in all_logs[1:]:
            log = merge_logfiles(log, nlg)
    return log


def merge_logfiles(log1, log2):
    """Merge two log files.

    If log1 overlaps with log2, the overlapping entries in log1
    are discarded.
    """
    first_in_2 = log2['time'][0]
    keep_from_1 = log1['time'] < first_in_2
    for key in log1.keys():
        log1[key] = log1[key][keep_from_1]
    log1.timeseries_append(log2)
    return log1


def results_summary(log, geolog, show=True):
    """3-panel summary plot from log and geoindex files
    """
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    okw = {'c':'purple', 'ls':'--', 'lw':1.5}
    mkw = {'c':'C0', 'ls':'-', 'lw':1.5}
    geolog.add_ae_quicklook(plot_obs=True, target=axes[0], val='AU', obs_kwargs=okw,
                            add_legend=False, **mkw)
    okw = {'c':'k', 'ls':'--', 'lw':1.5}
    mkw = {'c':'C1', 'ls':'-', 'lw':1.5}
    geolog.add_ae_quicklook(plot_obs=True, target=axes[0], val='AL', obs_kwargs=okw,
                            add_legend=False, **mkw)
    geolog.add_kp_quicklook(plot_obs=True, target=axes[1], add_legend=False)
    log.add_dst_quicklook(plot_obs=True, target=axes[2], add_legend=False)
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    axes[0].set_ylabel('AU/AL [nT]')
    axes[2].set_ylabel('Dst [nT]')
    from matplotlib.lines import Line2D

    legend_elements = [Line2D([0], [0], color='k', lw=1.5, label='Virtual (solid)'),
                       Line2D([0], [0], color='k', lw=1.5, ls='--', label='Observed (dashed)'),
                       ]
    axes[2].legend(handles=legend_elements, loc='lower left', ncol=2)
    if show:
        plt.show()
    return fig, axes

