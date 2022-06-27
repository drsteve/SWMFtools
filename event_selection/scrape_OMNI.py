
"""
Some exploratory code for scraping OMNIWeb with desired date ranges and variables.
Still in progress, need to select correct variables, format output year/day/hour/minute, visualize data.

@author: natalie klein
"""
# Standard Lib
import re
import datetime as dt
# Scientific Stack
import numpy as np
import seaborn as sns
sns.set(style="whitegrid", font_scale=1.5)
import matplotlib.pyplot as plt
import pandas as pd

# Third-party
import spacepy.datamodel as dm
from spacepy import pybats
import spacepy.plot as splot
import mechanize
try:
    import insitu_inference.code.bursty.readRC as readRC
except ImportError:
    # Local import for running from code directory
    import readRC


def make_imf_input(indata, fname=None, keylist=None):
    """Make an SWMF IMF input file from an input SpaceData"""
    if keylist is None:
        keylist = ['time', 'bx', 'by', 'bz', 'ux', 'uy', 'uz', 'rho', 'temp']
    numpts = indata['time'].shape[0]
    swmfdata = pybats.ImfInput(filename=False, load=False, npoints=numpts)
    naninds = set()
    for key in keylist:
        if key == 'time':
            # convert numpy datetime64 to python datetime
            swmfdata[key] = dm.dmarray(indata['time'].astype('M8[ms]').astype('O'))
        else:
            swmfdata[key] = dm.dmcopy(indata[key])
            naninds.update(np.where(np.isnan(swmfdata[key]))[0])
    # cull rows that have nan entries, SWMF will interpolate between
    # entries linearly. We can revisit this to use a noisy fill method if required
    naninds = np.array(sorted(list(naninds)))
    for key in keylist:
        swmfdata[key] = np.delete(swmfdata[key], naninds)
    swmfdata.attrs['coor'] = 'GSM'
    if fname is not None:
        swmfdata.write(fname)
    return swmfdata


def clean_OMNI(omarray, varname):
    """Given an array of OMNI data, replace the fill values with NaN
    """
    if 'velocity' in varname.lower() or varname in ['ux', 'uy', 'uz']:
        thr = 9999
    elif varname.startswith('B') or varname in ['bx', 'by', 'bz']:
        thr = 999
    elif 'density' in varname.lower() or varname.startswith('rho'):
        thr = 999
    elif 'temp' in varname.lower():
        thr = 9999999
    elif varname.lower().startswith('al') or varname.lower().startswith('sym'):
        thr = 99999
    elif 'pressure' in varname.lower() or varname in ['p_dyn']:
        thr = 99
    else:
        thr = None
    if isinstance(omarray, pd.Series) and thr is not None:
        omarray = omarray.mask(omarray >= thr)
    elif isinstance(omarray, dm.dmarray) and thr is not None:
        omarray = omarray.astype(float)
        omarray[omarray >= thr] = np.nan

    return omarray


def map_names(indata):
    """Change names of variables in record array"""
    # Copy names for iteration as we can't change something while we're
    # iterating over it
    fieldnames = dm.dmcopy(indata.dtype.names)
    newnames = []
    for name in fieldnames:
        if 'vx' in name.lower():
            newnames.append('ux')
        elif 'vy' in name.lower():
            newnames.append('uy')
        elif 'vz' in name.lower():
            newnames.append('uz')
        elif 'bx' in name.lower():
            newnames.append('bx')
        elif 'by' in name.lower():
            newnames.append('by')
        elif 'bz' in name.lower():
            newnames.append('bz')
        elif 'sym' in name.lower():
            newnames.append('sym-h')
        elif 'density' in name.lower():
            newnames.append('rho')
        elif 'temperature' in name.lower():
            newnames.append('temp')
        elif 'pressure' in name.lower():
            newnames.append('p_dyn')
        elif name == 'date':
            newnames.append('time')
        else:
            newnames.append(name.lower())

    indata.dtype.names = newnames

    return indata


def get_OMNI(start_date, end_date, as_pandas=True):
    """
    Retrieve OMNI data and format info from omniweb website

    Arguments
    ---------
    start_date : string
        Start date for data interval in YYMMDDHH format
    end_date : string
        End date for data interval in YYMMDDHH format
    as_pandas : boolean
        Set True to return pandas dataframe, False returns a SpaceData

    Returns
    -------
    dl_fmt : string
        Temporary filename for OMNI format description file
    dl : string
        Temporary filename for OMNI data file
    """
    br = mechanize.Browser()
    # set form variables for OMNIWeb
    br.open('https://omniweb.gsfc.nasa.gov/form/omni_min.html')
    br.select_form(name="frm")
    br['activity'] = ['ftp']
    br['res'] = ['min']
    br['start_date'] = start_date
    br['end_date'] = end_date
    br['vars'] = ['22', # Vx Velocity, GSE, km/s
                  '23', # Vy Velocity, GSE, km/s
                  '24', # Vz Velocity, GSE, km/s
                  '14', # Bx, GSM, nT
                  '17', # By, GSM, nT
                  '18', # Bz, GSM, nT
                  '25', # Proton Density, cm^{-3}
                  '26', # Proton Temperature, K
                  '27', # Flow pressure, nPa
                  '38', # AL index, nT
                  '41'] # Sym/H, nT

    # submit OMNIWeb form
    resp = br.submit()

    # Find links to download data and download it
    data_link = br.find_link(text_regex=re.compile(".lst")).url
    fmt_link = br.find_link(text_regex=re.compile(".fmt")).url
    dl = br.retrieve(data_link)[0]
    dl_fmt = br.retrieve(fmt_link)[0]

    # Read column names from .fmt file
    colnames = []
    with open(dl_fmt, 'r') as fh:
        fmt_info = fh.readlines()
    for line in fmt_info[4:]:
        colnames.append(' '.join(line.split()[1:-1]))
    
    # Read data from .lst file, format date
    df = pd.read_table(dl, header=None, delim_whitespace=True, names=colnames, parse_dates=[[0, 1, 2, 3]])
    df['date'] = df['Year_Day_Hour_Minute'].apply(lambda x: dt.datetime.strptime(x, '%Y %j %H %M'))
    if not as_pandas:
        recs = df.to_records()
        # map names to be more useful
        recs = map_names(recs)
        outdata = dm.fromRecArray(recs)
    else:
        outdata = df

    # Close browser to clean up temp files
    br.close()

    return outdata


def make_plot(plotrange, eventrange, df, show=True):
    """
    Make summary plot and either show (default) or save

    Arguments
    ---------
    plotrange : tuple of strings
        String timestamps for start and end of data interval
    eventrange : tuple of datetimes
        Datetimes marking iCME interval for highlighting
    """
    # TODO: update to use sharex and tidy time axis, labels, etc.
    start_date, end_date = plotrange
    start_datetime, end_datetime = eventrange
    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(12, 16))
    params_omni = ['BZ, nT (GSM)', 'Vx Velocity,km/s', 'Flow pressure, nPa', 'SYM/H, nT']
    params_sd = ['bz', 'ux', 'p_dyn', 'sym-h']
    labels = ['B$_{Z}^{GSM}$ [nT]', 'V$_{X}$ [km/s]', 'P$_{dyn}$', 'SYM-H [nT]']
    param = params_omni if isinstance(df, pd.DataFrame) else params_sd
    tvar = 'date' if isinstance(df, pd.DataFrame) else 'time'
    for par, cax, lab in zip(param, axes, labels):
        sns.lineplot(x=tvar, y=par, data=df, ax=cax)
        if tvar == 'time':
            cax.set_ylabel(lab)
        cax.axvspan(start_datetime, end_datetime, alpha=0.2, color='yellow')
        dolabel = True if par == 'sym-h' else False
        splot.applySmartTimeTicks(cax, df[tvar].astype('M8[ms]').astype('O'), dolabel=dolabel)
    fig.suptitle('RC event {} - {}'.format(start_date, end_date))
    if show:
        plt.show()
    else:
        plt.savefig('omni_rc_{}_{}.png'.format(start_date, end_date))
        plt.close()


if __name__ == '__main__':
    # User defined settings
    generate_plots = True  # Make summary plot for each event?
    show_plot = False  # True for display, False for save to file
    save_pickle = False  # Save a pickle of Pandas dataframes
    save_data = True  # Save data to HDF5 and SWMF ImfInput
    subset = 1#None  # Number of events (starting from end) -- None to do everything

    # default behavior is to use pandas and save to a pickle
    panda = True if save_pickle else False

    # Read Richardson-Cane data to get date ranges
    rc_data = readRC.read_list()
    n_events = len(rc_data['Epoch'])
    
    # Time period before/after event to extract
    td1 = dt.timedelta(hours=18) #12
    td2 = dt.timedelta(hours=12) #12

    # Empty list to store output data frames if save_pickle is True
    df_list = []

    # Loop over all event in the Richardson-Cane list
    st_at = 0 if subset is None else n_events-subset
    st_at = 103
    for rc_index in range(st_at, st_at + n_events):
        start_datetime = rc_data['ICME_start'][rc_index]
        end_datetime = rc_data['ICME_end'][rc_index]
        start_date = (start_datetime - td1).strftime('%Y%m%d%H')
        end_date = (end_datetime + td2).strftime('%Y%m%d%H')
    
        # Save data (.lst file) and data format (.fmt file)
        df = get_OMNI(start_date, end_date, as_pandas=panda)
        # And mask fill
        colnames = df.columns if panda else df.keys()
        for colname in colnames:
            df[colname] = clean_OMNI(df[colname], colname)

        # process depending on pandas or spacedata output
        if not panda:
            df.attrs['event_id'] = rc_index
        else:
            df['event_id'] = rc_index

        # Save per user request
        if save_pickle:
            df_list.append(df)
        elif save_data:
            df.toHDF5('omni_rc_event{:03d}.h5'.format(rc_index))
            swmf = make_imf_input(df, fname='IMF_rc{:03d}_{}.dat'.format(rc_index, start_date))

        if generate_plots:
            fig = make_plot((start_date, end_date), (start_datetime, end_datetime), df, show=show_plot)

        print('Finished RC event %d of %d' % (rc_index, n_events-1))
    
    if save_pickle:
        all_df = pd.concat(df_list)
        all_df.to_pickle('OMNI_rc_dataframe.pkl')
