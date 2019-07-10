#stdlib
import os
import ftplib
import itertools
import argparse
import datetime as dt
#scientific stack
import numpy as np
import scipy
import scipy.interpolate
import spacepy.datamodel as dm
import spacepy.time as spt
import spacepy.pybats as swmf
from sklearn.neighbors import KernelDensity
#local
import perturbplot as pplot


def get_SC_OMNI(year=2000, bird='ACE', datadir='Data', force=False, verbose=False, **kwargs):
    '''Download S/C specific OMNI file'''
    valid_birds = ['ACE', 'IMP', 'GEOTAIL', 'WIND']
    if bird.upper() not in valid_birds:
        raise ValueError('Invalid satellite selected ({0})'.format(bird))
    targ_fn = '{0}_min_b{1}.txt'.format(bird.lower(), year)
    #now check whether we have this file already
    if not force and os.path.isfile(os.path.join(datadir, targ_fn)):
        if verbose: print('Data already present for {0} in {1} - not downloading'.format(bird, year))
        return os.path.join(datadir, targ_fn)
    #now download the file and save in datadir
    omni_ftp = 'spdf.gsfc.nasa.gov'
    sc_dir = 'pub/data/omni/high_res_omni/sc_specific/'
    ftp = ftplib.FTP(omni_ftp)
    ftp.login()
    ftp.cwd(sc_dir)
    with open(os.path.join(datadir, targ_fn), 'w') as ofh:
        ftp.retrlines('RETR {0}'.format(targ_fn), lambda s, w = ofh.write: w(s + '\n'))
    print('Retrieved {0}'.format(targ_fn))
    return os.path.join(datadir, targ_fn)


def load_SC_OMNI(bird, year, outdata=None, **kwargs):
    '''Load satellite specific OMNI data into dict'''
    fname = get_SC_OMNI(year=year, bird=bird, **kwargs)
    dum = np.genfromtxt(fname, usecols=(0,1,2,3,15,16,23,26,28,29,30), 
                         names=('year','day','hour','minute','By_GSM','Bz_GSM','Vx_GSE','Den_P','X_GSE','Y_GSE','Z_GSE'),
                         converters={0: int, 1: int, 2: int, 3: int})
    data = dm.fromRecArray(dum)
    dates = spt.doy2date(data['year'], data['day'], dtobj=True)
    times = [dt.timedelta(hours=x, minutes=y) for x,y in zip(data['hour'],data['minute'])]
    data['DateTime'] = dates + times
    for key in ['year', 'day', 'hour', 'minute']:
        del data[key]
    data['Bz_GSM'][np.abs(data['Bz_GSM'])>20] = np.nan
    data['By_GSM'][np.abs(data['By_GSM'])>20] = np.nan
    data['Vx_GSE'][np.abs(data['Vx_GSE'])>900] = np.nan
    data['X_GSE'][np.abs(data['X_GSE'])>9000] = np.nan
    data['Y_GSE'][np.abs(data['Y_GSE'])>9000] = np.nan
    data['Z_GSE'][np.abs(data['Z_GSE'])>9000] = np.nan
    if outdata:
        for key in ['By_GSM', 'Bz_GSM', 'Vx_GSE', 'DateTime', 'Den_P', 'X_GSE', 'Y_GSE', 'Z_GSE']:
            outdata[key] = np.concatenate([outdata[key], data[key]])
        return outdata
    return data



if __name__=='__main__':
    #python perturbSWMF.py -p Nov2003 -f IMF.dat -n 6 -s 1977
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', dest='seed', type=int, default=1234, help='Specify random seed. Integer. Default=1234')
    parser.add_argument('-n', '--number', dest='Nensembles', type=int, default=8, help='Number of perturbed files to generate. Default=8')
    parser.add_argument('-f', '--file', dest='fname', default='IMF_ev5.dat', help='Input SWMF IMF filename. Default "IMF_ev5.dat"')
    parser.add_argument('-p', '--path', dest='path', default='SWMF_inputs', help='Path for input/output')
    options = parser.parse_args()

    np.random.seed(options.seed) #set seed for repeatability

    #read SWMF ImfInput file
    infilename = os.path.join(options.path, options.fname)
    if os.path.isfile(infilename):
        eventIMF = swmf.ImfInput(filename=infilename)
    else:
        raise IOError('Specified input file does not appear to exist or is not readable')

    Ntimes         = len(eventIMF['ux']) #3*1440 #N days at 1-min resolution
    generateInputs = True
    saveErrors     = False

    varlist = ['Vx_GSE', 'Bz_GSM', 'By_GSM']
    Nvars = len(varlist)
    map_dict = {'Vx_GSE': 'ux',
                'Bz_GSM': 'bz',
                'By_GSM': 'by'}
    ylimdict = {'Vx_GSE': [-300, -800],
                'Bz_GSM': [-20, 20],
                'By_GSM': [-20, 20]}
    xlimdict = {'Vx_GSE': [-60, 60],
                'Bz_GSM': [-15, 15],
                'By_GSM': [-15, 15]}
    unitsdict = {'Vx_GSE': '[km/s]',
                'Bz_GSM': '[nT]',
                'By_GSM': '[nT]'}
    
    #load ACE data into dict (ups: upstream)
    upsdata = load_SC_OMNI('ace', 1999)
    upsdata = load_SC_OMNI('ace', 2000, outdata=upsdata)
    upsdata = load_SC_OMNI('ace', 2001, outdata=upsdata)
    upsdata = load_SC_OMNI('ace', 2002, outdata=upsdata)
    upsdata = load_SC_OMNI('ace', 2003, outdata=upsdata)
    upsdata = load_SC_OMNI('ace', 2004, outdata=upsdata)
    upsdata = load_SC_OMNI('ace', 2005, outdata=upsdata)

    #load GEOTAIL data into dict (nmp: near magnetopause)
    nmpdata = load_SC_OMNI('geotail', 1999)
    nmpdata = load_SC_OMNI('geotail', 2000, outdata=nmpdata)
    nmpdata = load_SC_OMNI('geotail', 2001, outdata=nmpdata)
    nmpdata = load_SC_OMNI('geotail', 2002, outdata=nmpdata)
    nmpdata = load_SC_OMNI('geotail', 2003, outdata=nmpdata)
    nmpdata = load_SC_OMNI('geotail', 2004, outdata=nmpdata)
    nmpdata = load_SC_OMNI('geotail', 2005, outdata=nmpdata)

    print(nmpdata['DateTime'][0], nmpdata['DateTime'][-1])
    savedata = dm.SpaceData()
    for var in varlist[::-1]:
        print('Processing {}'.format(var))
        err = 'epsilon'
        varlabel = var[0]+'$_'+var[1]+'$'
        errlabel = r'$\varepsilon$'

        plotinfo = {'var': var,
                    'err': err,
                    'varlabel': varlabel,
                    'errlabel': errlabel,
                    'xlimdict': xlimdict,
                    'ylimdict': ylimdict,
                    'units': unitsdict}

        #Get error distrib for var as fn of var and plot
        valid_inds = np.logical_and(np.isfinite(nmpdata[var]), np.isfinite(upsdata[var]))
        err = nmpdata[var]-upsdata[var]
        errors = err[valid_inds]
        savedata[var] = errors

    #generate error series with block resampling (cf. moving block bootstrap)
    #use
    error_series = np.empty([options.Nensembles, Ntimes, Nvars])
    blocksize = 60
    n_blocks = 1 + Ntimes//blocksize
    for run_num in range(options.Nensembles):
        #rather than building a 3D array here I should modify an SWMF input file directly
        blockstarts = np.random.randint(0, len(errors)-blocksize, n_blocks)
        for it, bidx in enumerate(blockstarts):
            if Ntimes-it*blocksize>blocksize:
                for vidx, var in enumerate(varlist):
                    error_series[run_num, it*blocksize:it*blocksize+blocksize, vidx] = savedata[var][bidx:bidx+blocksize]
            elif Ntimes-it*blocksize>0:
                room = len(error_series[run_num, it*blocksize:, vidx])
                error_series[run_num, it*blocksize:, vidx] = savedata[var][bidx:bidx+room]
            else:
                pass
        #modify SWMF ImfInput and write new file
        outfilename = '.'.join(['_'.join([infilename.split('.')[0],'{0:03d}'.format(run_num)]), 'dat'])
        if generateInputs:
            surrogateIMF = dm.dmcopy(eventIMF)
            for vidx, var in enumerate(varlist):

                surrogateIMF[map_dict[var]] += error_series[run_num, :Ntimes, vidx]
            #then write to file
            surrogateIMF.write(outfilename)


    #save error series if req'd
    if saveErrors:
        out = dm.SpaceData()
        out['errors'] = dm.dmarray(error_series)
        out['errors'].attrs['DEPEND_0'] = 'EnsembleNumber'
        out['errors'].attrs['DEPEND_1'] = 'Timestep'
        out['errors'].attrs['DEPEND_2'] = 'Variable'
        out.toHDF5('MBB_errors.h5'.format(var))
