import glob
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import spacepy.datamodel as dm
import spacepy.time as spt
import spacepy.toolbox as tb
import seaborn as sns
import pandas as pd


def autoevents(data, n_events=5, bin1d=4, seed=None):
    """Select events using equal-weight-bin sampling method

    N-d parameter space is broken into a K**N element grid.
    Any bins with no samples are discarded.
    All other bins are weighted equally.
    Each requested event is selected by first choosing a
    random bin (without replacement) and then randomly choosing
    one event from that bin.
    The selected events are therefore distributed across the
    parameter space without being weighted to the probability of
    event occurrence within the region of parameter space.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with parameters (each row is one event, each
        column is a parameter)
    n_events : int
        Number of events to select
    bin1d : int
        Number of bins to use in each dimension
    seed : int (or None)
        Random seed. Default None
    """
    # Set random seed for reproducible results
    if seed:
        np.random.seed(seed)
    # Standardize all variables and drop events with NaN values
    standardize = lambda x: (x - x.mean())/x.std()
    filt = dframe.dropna()
    try:
        filt = filt.drop(columns=['max(E_sw)'])
    except:
        pass
    filt = filt.apply(standardize)
    ndims = len(filt.columns)
    grid = dict()
    inds = np.empty((filt.shape[0], filt.shape[1])).astype(int)
    # Loop over variables, get limits, set grid
    for var in filt.columns:
        maxval = filt[var].max()
        minval = filt[var].min()
        maxsign = np.sign(maxval)
        roundup = np.ceil(np.abs(maxval)) if maxval > 0 else np.floor(np.abs(maxval))
        roundup *= maxsign
        minsign = np.sign(minval)
        rounddn = np.ceil(np.abs(minval)) if minval < 0 else np.floor(np.abs(minval))
        rounddn *= minsign
        grid[var] = np.linspace(rounddn, roundup, num=bin1d)
        # Now find the bin in this dimension for all events
        inds[:, filt.columns.get_loc(var)] = np.digitize(filt[var], grid[var])
    # Finf "flattened" bin for each event and whether bins are used
    cells = set()
    binned_data = dict()
    # For each event ...
    for idx, row in enumerate(inds):
        # Find the flattened index
        bin_num = np.ravel_multi_index(row, [bin1d]*ndims)
        cells.add(bin_num)  # Make sure we add this to the set of bins with data
        if bin_num not in binned_data:
            binned_data[bin_num] = []
        binned_data[bin_num].append(idx) # And add the event number to the bin
    cells = np.asarray(list(cells))
    # Choose cells to sample from
    use_cells = np.random.choice(cells, n_events, replace=False)
    # from each cell, find the events and pick one
    use_events = [np.random.choice(binned_data[ind], 1)[0] for ind in use_cells]
    return use_events


makeNew = False

if makeNew:
    # read HDF5 data for each event
    # ensure it's sorted so the event numbers have a chance of lining up
    allh5 = sorted(glob.glob('*.h5'))
    
    # Gather event summary stats
    symh = []
    al   = []
    vx   = []
    esw  = []
    bzhr = []
    evno = []
    stda = []
    
    for idx, h5fn in enumerate(allh5):
        if idx%10 == 0: print('Reading file {} of {}'.format(idx+1, len(allh5)))
        data = dm.fromHDF5(h5fn)
        data['time'] = spt.Ticktock(data['time']).UTC
        symh.append(np.nanmin(np.asarray(data['sym-h'])))
        al.append(np.nanmin(np.asarray(data['al-index, nt'])))
        umask = np.isfinite(data['ux'])
        uxmin = np.nan if data['ux'][umask].size==0 else np.nanmin(data['ux'][umask])
        vx.append(uxmin)
        ey = 1e3 * data['ux'] * data['bz']
        emask = np.isfinite(ey)
        ey[emask][ey[emask]<0] = 0
        eymax = np.nan if ey[emask].size==0 else ey[emask].max()
        esw.append(eymax)
        # hourly avgs of bz
        outd, outt = tb.windowMean(data['bz'], time=data['time'], winsize=dt.timedelta(hours=1),
                                   overlap=dt.timedelta(0), st_time = data['time'][0].replace(minute=0), op=np.nanmean)
        bzhr.append(np.nanmin(outd))
        evno.append('{:03d}'.format(idx))
        stda.append(data['time'][0].strftime('%Y%m%d%H'))
    
    # make dataframe to pass to seaborn pairplot
    # go via dict-like
    summary = dm.SpaceData()
    summary['min(Sym-H)'] = dm.dmarray(symh)
    summary['min(AL)'] = dm.dmarray(al)
    summary['min(Vx)'] = dm.dmarray(vx)
    summary['max(E_sw)'] = dm.dmarray(esw)
    summary['min(<Bz>)'] = dm.dmarray(bzhr)
    summary['eventNumber'] = dm.dmarray(evno)
    summary['startTime'] = dm.dmarray(stda)
    
    # write summary values and event ID to ASCII
    dm.toJSONheadedASCII('rc_event_summary.txt', summary, order=['eventNumber', 'min(Sym-H)',
                                                                 'min(AL)', 'min(Vx)', 'max(E_sw)',
                                                                 'min(<Bz>)'])
    del summary['eventNumber']
    del summary['startTime']
else:
    tmp = dm.readJSONheadedASCII('rc_event_summary.txt')
    summary = dm.SpaceData()
    #summary['min(Sym-H)'] = tmp['min(Sym-H)']
    summary['min(AL)'] = tmp['min(AL)']
    summary['min(Vx)'] = tmp['min(Vx)']
    #summary['max(E_sw)'] = tmp['max(E_sw)']
    summary['min(<Bz>)'] = tmp['min(<Bz>)']

# pair plots
if True:
    dframe = pd.DataFrame(summary)
    seed = 1405
    selections = [#('SKM', [57, 103, 156, 259, 443], 'maroon', 'd'),
                  #('AB2', [47, 124, 240, 266, 360], 'navy', '*'),
                  #('ECL', [78, 103, 200, 240, 284], 'gold', 'x'),
                  #('AB1', [38, 79, 98, 184, 207], 'black', '2'),
                  #('NEK', [152, 253, 258, 266, 295], 'orchid', '.'),
                  ('Auto', autoevents(dframe, n_events=15, bin1d=5, seed=seed), 'maroon', 'x'),
                  ]
    # Now pass this through to seaborn PairGrid, as it's much more flexible than pairplot
    grid = sns.PairGrid(dframe, height=1.75)
    grid = grid.map_diag(plt.hist, bins='auto')
    grid = grid.map_lower(plt.scatter, marker='.', s=5, color='seagreen')
    grid = grid.map_upper(sns.kdeplot, n_levels=25, cmap='YlGnBu', shade=True, shade_lowest=False)
    axes = grid.axes
    for name, sel, rgb, shape in selections:
        pkw = {'color': rgb, 'linestyle': 'none', 'marker': shape}
        #axes[1][0].plot(summary['min(Sym-H)'][sel], summary['min(AL)'][sel], **pkw)
        #axes[2][0].plot(summary['min(Sym-H)'][sel], summary['min(Vx)'][sel], **pkw)
        #axes[2][1].plot(summary['min(AL)'][sel], summary['min(Vx)'][sel], **pkw)
        #axes[3][0].plot(summary['min(Sym-H)'][sel], summary['max(E_sw)'][sel], **pkw)
        #axes[3][1].plot(summary['min(AL)'][sel], summary['max(E_sw)'][sel], **pkw)
        #axes[3][2].plot(summary['min(Vx)'][sel], summary['max(E_sw)'][sel], **pkw)
        #axes[4][0].plot(summary['min(Sym-H)'][sel], summary['min(<Bz>)'][sel], **pkw)
        #axes[4][1].plot(summary['min(AL)'][sel], summary['min(<Bz>)'][sel], **pkw)
        #axes[4][2].plot(summary['min(Vx)'][sel], summary['min(<Bz>)'][sel], **pkw)
        #axes[4][3].plot(summary['max(E_sw)'][sel], summary['min(<Bz>)'][sel], **pkw)
        axes[1][0].plot(summary['min(AL)'][sel], summary['min(Vx)'][sel], **pkw)
        axes[2][0].plot(summary['min(AL)'][sel], summary['min(<Bz>)'][sel], **pkw)
        axes[2][1].plot(summary['min(Vx)'][sel], summary['min(<Bz>)'][sel], **pkw)
    # grid.fig.suptitle('{} selection\nEvents {}, {}, {}, {}, {}'.format(name, *select))
    plt.tight_layout()
    #plt.savefig('RC_pairsplot_pointscomparison_AUTO.png', dpi=300)
    plt.savefig('RC_pairsplot_piyush_{}.png'.format(seed), dpi=300)
    plt.close('all')
