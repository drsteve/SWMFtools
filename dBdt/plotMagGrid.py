import glob
import re
import os
import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shapereader
import spacepy.datamodel as dm
from spacepy.pybats import bats

import gmdtools


def ensembleThresh(searchpatt='mag_grid_e20100405-083[12][0-5]0.out', outdir='dBdt_images'):
    useProject = ccrs.PlateCarree #Projection to use for plotting, e.g., ccrs.AlbersEqualArea
    plotprojection = useProject(central_longitude=-90.0)

    pdata = {'thresh': 2.0, #nT/s
             'plotvar': 'dBdth',
             'dataprojection': ccrs.PlateCarree(),
             'plotprojection': plotprojection,
            }

    #add transmission lines from ArcGIS shapefile
    fname = 'txlines/Transmission_lines.shp'
    txshapes = list(shapereader.Reader(fname).geometries())
    albers = ccrs.AlbersEqualArea(central_longitude=-96.0,
                                  central_latitude=23.0,
                                  standard_parallels=(29.5,45.5))
    txinpc = [plotprojection.project_geometry(ii, albers) for ii in txshapes]
    pdata['shapes'] = txinpc

    members = {'run_fullmag_001': [],
               'run_fullmag_002': [],
               'run_fullmag_030': [],
               'run_fullmag_042': [],
               #'run_fullmag_043': [],
               'run_fullmag_044': [],
               }
    colors = {'run_fullmag_001': 'firebrick',
              'run_fullmag_002': 'goldenrod',
              'run_fullmag_030': 'olive',
              'run_fullmag_042': 'skyblue',
              #'run_fullmag_043': 'slateblue',
              'run_fullmag_044': 'brown',
              }
    for key in members.keys():
        rundir = key[4:]
        globterm = os.path.join(key, 'RESULTS', rundir, 'GM', searchpatt)
        members[key] = sorted(glob.glob(globterm))
    tstep=10 #diff between subsequent files in seconds
    allruns = list(members.keys())
    #following code assumes that same files are found by glob for each ensemble
    #member, so we'll check that here to avoid mixing timesteps
    tmplist = [os.path.basename(fn) for fn in members[allruns[0]]]
    for memb in allruns[1:]:
        complist = [os.path.basename(fn) for fn in members[memb]]
        try:
            assert tmplist==complist
        except AssertionError:
            print(memb)
            raise AssertionError

    #startpoint, forward diff.
    baseline = allruns.pop()
    infiles = members[baseline]
    pdata['ccolor'] = colors[baseline]

    mdata = bats.MagGridFile(infiles[0])
    plusone = bats.MagGridFile(infiles[1])
    plustwo = bats.MagGridFile(infiles[2])
    mdata['dBdtn'] = dm.dmarray(forwardDiff(mdata['dBn'], plusone['dBn'],
                                            plustwo['dBn'], tstep),
                                            attrs={'units': 'nT/s'})
    mdata['dBdte'] = dm.dmarray(forwardDiff(mdata['dBe'], plusone['dBe'],
                                            plustwo['dBe'], tstep),
                                            attrs={'units': 'nT/s'})
    mdata['dBdth'] = dm.dmarray(np.sqrt(mdata['dBdtn']**2 +
                                        mdata['dBdte']**2),
                                        attrs={'units': 'nT/s'})
    ax = gmdtools.plotContour(mdata, pdata, addTo=None)

    for memb in allruns:
        mfiles = members[memb]
        pdata['ccolor'] = colors[memb]
        mdata = bats.MagGridFile(mfiles[0])
        plusone = bats.MagGridFile(mfiles[1])
        plustwo = bats.MagGridFile(mfiles[2])
        mdata['dBdtn'] = dm.dmarray(forwardDiff(mdata['dBn'], plusone['dBn'],
                                                plustwo['dBn'], tstep),
                                                attrs={'units': 'nT/s'})
        mdata['dBdte'] = dm.dmarray(forwardDiff(mdata['dBe'], plusone['dBe'],
                                                plustwo['dBe'], tstep),
                                                attrs={'units': 'nT/s'})
        mdata['dBdth'] = dm.dmarray(np.sqrt(mdata['dBdtn']**2 +
                                            mdata['dBdte']**2),
                                            attrs={'units': 'nT/s'})
        ax = gmdtools.plotContour(mdata, pdata, addTo=ax)
    #windows can't handle colons in filenames...
    isotime = mdata.attrs['time'].isoformat()
    plt.savefig(os.path.join(outdir, r'{0}_{1}.png'.format(pdata['plotvar'],
                             isotime.replace(':',''))), dpi=300)
    plt.close()

    #all except endpoints, central diff
    for idx, fname in enumerate(infiles[1:-1], start=1):
        pdata['ccolor'] = colors[baseline]
        minusone = bats.MagGridFile(infiles[idx-1])
        mdata = bats.MagGridFile(fname)
        plusone = bats.MagGridFile(infiles[idx+1])
        mdata['dBdtn'] = dm.dmarray(centralDiff(minusone['dBn'],
                                                plusone['dBn'], tstep),
                                                attrs={'units': 'nT/s'})
        mdata['dBdte'] = dm.dmarray(centralDiff(minusone['dBe'],
                                                plusone['dBe'], tstep),
                                                attrs={'units': 'nT/s'})
        mdata['dBdth'] = dm.dmarray(np.sqrt(mdata['dBdtn']**2 +
                                            mdata['dBdte']**2),
                                            attrs={'units': 'nT/s'})
        ax = plotContour(mdata, pdata, addTo=None)
        for memb in allruns:
            mfiles = members[memb]
            pdata['ccolor'] = colors[memb]
            minusone = bats.MagGridFile(mfiles[idx-1])
            mdata = bats.MagGridFile(mfiles[idx])
            plusone = bats.MagGridFile(mfiles[idx+1])
            mdata['dBdtn'] = dm.dmarray(centralDiff(minusone['dBn'],
                                                    plusone['dBn'], tstep),
                                                    attrs={'units': 'nT/s'})
            mdata['dBdte'] = dm.dmarray(centralDiff(minusone['dBe'],
                                                    plusone['dBe'], tstep),
                                                    attrs={'units': 'nT/s'})
            mdata['dBdth'] = dm.dmarray(np.sqrt(mdata['dBdtn']**2 +
                                                mdata['dBdte']**2),
                                                attrs={'units': 'nT/s'})
            ax = gmdtools.plotContour(mdata, pdata, addTo=ax)
        #windows can't handle colons in filenames...
        isotime = mdata.attrs['time'].isoformat()
        plt.savefig(os.path.join(outdir, r'{0}_{1}.png'.format(pdata['plotvar'],
                                 isotime.replace(':',''))), dpi=300)
        plt.close()

    #final point, backwards diff.
    pdata['ccolor'] = colors[baseline]
    minustwo = bats.MagGridFile(infiles[-3])
    minusone = bats.MagGridFile(infiles[-2])
    mdata = bats.MagGridFile(infiles[-1])
    mdata['dBdtn'] = dm.dmarray(backwardsDiff(minustwo['dBn'], minusone['dBn'],
                                              mdata['dBn'], tstep),
                                              attrs={'units': 'nT/s'})
    mdata['dBdte'] = dm.dmarray(backwardsDiff(minustwo['dBn'], minusone['dBe'],
                                              mdata['dBe'], tstep),
                                              attrs={'units': 'nT/s'})
    mdata['dBdth'] = dm.dmarray(np.sqrt(mdata['dBdtn']**2 + mdata['dBdte']**2),
                                attrs={'units': 'nT/s'})
    ax = gmdtools.plotContour(mdata, pdata, addTo=None)
    for memb in allruns:
        mfiles = members[memb]
        pdata['ccolor'] = colors[memb]
        minustwo = bats.MagGridFile(mfiles[-3])
        minusone = bats.MagGridFile(mfiles[-2])
        mdata = bats.MagGridFile(mfiles[-1])
        mdata['dBdtn'] = dm.dmarray(backwardsDiff(minustwo['dBn'],
                                                  minusone['dBn'],
                                                  mdata['dBn'], tstep),
                                                  attrs={'units': 'nT/s'})
        mdata['dBdte'] = dm.dmarray(backwardsDiff(minustwo['dBe'],
                                                  minusone['dBe'],
                                                  mdata['dBn'], tstep),
                                                  attrs={'units': 'nT/s'})
        mdata['dBdth'] = dm.dmarray(np.sqrt(mdata['dBdtn']**2 +
                                            mdata['dBdte']**2),
                                            attrs={'units': 'nT/s'})
        ax = gmdtools.plotContour(mdata, pdata, addTo=ax)
    #windows can't handle colons in filenames...
    isotime = mdata.attrs['time'].isoformat()
    plt.savefig(os.path.join(outdir, r'{0}_{1}.png'.format(pdata['plotvar'],
                             isotime.replace(':',''))), dpi=300)
    plt.close()

    #make symlinks to images with numeric naming so it can be easily passed to ffmpeg
    gmdtools.makeSymlinks(outdir, kind='dBdt')


def singleRundBdt(runname, searchpatt='mag_grid_e20100405-0[89][0-5][0-9][03]0.out',
                  outdir = 'dBdt_maps', links=True):
    useProject = ccrs.PlateCarree #Projection to use for plotting, e.g., ccrs.AlbersEqualArea
    plotprojection = useProject(central_longitude=-90.0)

    pdata = {'plotvar': 'dBdth',
             'dataprojection': ccrs.PlateCarree(),
             'plotprojection': plotprojection,
            }

    #add transmission lines from ArcGIS shapefile
    #fname = 'txlines/Transmission_lines.shp'
    #txshapes = list(shapereader.Reader(fname).geometries())
    #albers = ccrs.AlbersEqualArea(central_longitude=-96.0,
    #                              central_latitude=23.0,
    #                              standard_parallels=(29.5,45.5))
    #txinpc = [plotprojection.project_geometry(ii, albers) for ii in txshapes]
    pdata['shapes'] = None#txinpc

    rundir = runname[4:]
    globterm = os.path.join(runname, 'RESULTS', rundir, 'GM', searchpatt)
    globterm = os.path.join(runname, searchpatt)
    tstep=60#30 #diff between subsequent files in seconds
    allfiles = sorted(glob.glob(globterm))

    #startpoint, forward diff.
    infiles = allfiles

    #all except endpoints, central diff
    for idx, fname in enumerate(infiles[1:-1], start=1):
        minusone = bats.MagGridFile(infiles[idx-1])#, format='ascii')
        mdata = bats.MagGridFile(fname)#, format='ascii')
        plusone = bats.MagGridFile(infiles[idx+1])#, format='ascii')
        mdata['dBdtn'] = dm.dmarray(gmdtools.centralDiff(minusone['dBn'],
                                                plusone['dBn'], tstep),
                                                attrs={'units': 'nT/s'})
        mdata['dBdte'] = dm.dmarray(gmdtools.centralDiff(minusone['dBe'],
                                                plusone['dBe'], tstep),
                                                attrs={'units': 'nT/s'})
        mdata['dBdth'] = dm.dmarray(np.sqrt(mdata['dBdtn']**2 +
                                            mdata['dBdte']**2),
                                            attrs={'units': 'nT/s'})
        ax = gmdtools.plotFilledContours(mdata, pdata, addTo=None)
        #windows can't handle colons in filenames...
        isotime = mdata.attrs['time'].isoformat()
        plt.savefig(os.path.join(outdir, r'{0}_{1}.png'.format(pdata['plotvar'],
                                 isotime.replace(':',''))), dpi=300)
        plt.close()
    if links:
        gmdtools.makeSymlinks(outdir, kind='dBdt')


def singleRunE(runname, searchpatt='1_geoe_*.csv', outdir = 'E_maps', vecs=True, links=True):
    useProject = ccrs.PlateCarree #Projection to use for plotting, e.g., ccrs.AlbersEqualArea
    plotprojection = useProject(central_longitude=-90.0)

    pdata = {'plotvar': 'Emag',
             'plotvec': 'E',
             'dataprojection': ccrs.PlateCarree(),
             'plotprojection': plotprojection,
            }

    #add transmission lines from ArcGIS shapefile
    #fname = 'txlines/Transmission_lines.shp'
    #txshapes = list(shapereader.Reader(fname).geometries())
    #albers = ccrs.AlbersEqualArea(central_longitude=-96.0,
    #                              central_latitude=23.0,
    #                              standard_parallels=(29.5,45.5))
    #txinpc = [plotprojection.project_geometry(ii, albers) for ii in txshapes]
    pdata['shapes'] = None#txinpc

    rundir = runname[4:]
    globterm = os.path.join(runname, searchpatt)
    allfiles = sorted(glob.glob(globterm))
    infiles = allfiles
    #downselect files to use
    #TODO: remove hardcoded downselect of timerange
    #allfiles = [fn for fn in allfiles if (int(re.search('\d{8}-(\d{6})', fn).groups()[0]) >= 80000) and (int(re.search('\d{8}-(\d{6})', fn).groups()[0]) <= 90000)]
    #infiles = [fn for fn in allfiles if re.search('\d{8}-(\d{6})', fn).groups()[0][-2:] in ['00', '15', '30', '45']]

    #loop over files and plot map of |E|
    for fname in infiles:
        edata = gmdtools.readEcsv(fname)
        ax = gmdtools.plotFilledContours(edata, pdata, addTo=None)
        runnum = int(rundir[-3:])
        if vecs:
            ax = gmdtools.plotVectors(edata, pdata, addTo=ax, quiver=False, maxVec=2.75)
            anchor = (-9.195, 38.744) #Lisbon #(-7.9304, 37.0194) #Faro, Port.
            ax.text(anchor[0], anchor[1], 'Member #{0}'.format(runnum), verticalalignment='top', weight='semibold',
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='None'), transform=ccrs.PlateCarree())
        #windows can't handle colons in filenames...
        isotime = edata.attrs['time'].isoformat()
        if 'QBC' in rundir: runnum = 'QBC{0}'.format(runnum)
        plt.savefig(os.path.join(outdir, r'{2}_{0}_{1}.png'.format(pdata['plotvar'],
                                 isotime.replace(':',''), runnum)), dpi=300)
        plt.close()
    if links:
        gmdtools.makeSymlinks(outdir, kind='E')


def singleRunEwithEnsemble(runname, ensname, searchpatt='1_geoe_*.csv', outdir='E_maps', eVar='Estd'):
    '''
    Optional Parameters
    ------------------

    eVar : string
        Estd or Emag
    '''
    import cartopy.feature as cfea
    import cartopy.feature.nightshade as night
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    useProject = ccrs.PlateCarree #Projection to use for plotting, e.g., ccrs.AlbersEqualArea
    plotprojection = useProject(central_longitude=-90.0)

    pdata = {'plotvar': 'Emag',
             'plotvec': 'E',
             'dataprojection': ccrs.PlateCarree(),
             'plotprojection': plotprojection,
            }
    doQuiver = False #may want to change this in future, so I'm leaving the code in
    #add features from ArcGIS shapefile?
    pdata['shapes'] = None

    rundir = runname[4:]
    globterm = os.path.join(runname, searchpatt)
    allfiles = sorted(glob.glob(globterm))

    #downselect files to use
    #TODO: remove hardcoded downselect of timerange
    allfiles = [fn for fn in allfiles if (int(re.search('\d{8}-(\d{6})', fn).groups()[0]) >= 81500) and (int(re.search('\d{8}-(\d{6})', fn).groups()[0]) <= 84500)]
    infiles = [fn for fn in allfiles if re.search('\d{8}-(\d{6})', fn).groups()[0][-2:] in ['00', '15', '30', '45']]

    #get ensemble member directories
    members = [en for en in glob.glob(ensname) if en!=runname]
    print('Using {0} ensemble members'.format(len(members)))
    for mm in members:
        print('{0}'.format(mm))

    #loop over files and plot map of |E|
    for fname in infiles:
        edata = gmdtools.readEcsv(fname)
        #find matching file for each ens. member
        filepart = os.path.split(fname)[-1]
        collect = [edata]
        for mdir in members:
            fname = os.path.join(mdir, filepart)
            try:
                collect.append(gmdtools.readEcsv(fname))
            except:
                print('Hit an issue with {0}'.format(fname))
                continue
        #collect Emag and calculate stadard deviation across ensemble members at each gridpoint
        combinedMag = np.dstack(cc['Emag'] for cc in collect)
        combinedEast = np.dstack(cc['Ee'] for cc in collect)
        combinedNorth = np.dstack(cc['En'] for cc in collect)
        ensmean = dm.dmcopy(edata)
        ensmean['Ee_raw'] = np.stack(cc['Ee_raw'] for cc in collect).mean(axis=0)
        ensmean['En_raw'] = np.stack(cc['En_raw'] for cc in collect).mean(axis=0)
        ensmean['Estd'] = combinedMag.std(axis=-1)
        ensmean['Ee_ensMean'] = combinedEast.mean(axis=-1)
        ensmean['En_ensMean'] = combinedNorth.mean(axis=-1)
        ensmean['Emag'] = np.sqrt(ensmean['Ee_ensMean']**2 + ensmean['En_ensMean']**2)
        #
        print('Working on timestep {0}. # members = {1}'.format(edata.attrs['time'], len(collect)))
        #
        fig = plt.figure(figsize=(10,5.5))
        #set up first map panel (reference run)
        pdata['plotvar'] = 'Emag'
        sstyle = {'color': 'black'}
        qstyle = {'color': 'darkgrey', 'pivot': 'mid', 'alpha': 0.6, 'scale': 1, 'scale_units': 'xy'}
        ax = plt.subplot(2, 1, 1, projection=pdata['plotprojection'])
        ax.coastlines(color='darkgrey')
        ax.add_feature(cfea.BORDERS, linestyle=':', edgecolor='darkgrey')
        ax.add_feature(night.Nightshade(edata.attrs['time'], alpha=0.3))
        lon_formatter = LongitudeFormatter(number_format='.1f',
                                           degree_symbol='',
                                           dateline_direction_label=True)
        lat_formatter = LatitudeFormatter(number_format='.1f',
                                          degree_symbol='')
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.set_extent([-165.0, 45.0, 30.0, 80.0], crs=ccrs.PlateCarree())
        ax = gmdtools.plotFilledContours(edata, pdata, addTo=ax)
        ax = gmdtools.plotVectors(edata, pdata, addTo=ax, maxVec=2.75, sstyle=sstyle)
        ax.set_title('{0}'.format(edata.attrs['time'].isoformat()))
        #set up second map panel (use requested (Emag_ens or Estd)
        ax2 = plt.subplot(2, 1, 2, projection=pdata['plotprojection'])
        ax2.coastlines(color='darkgrey')
        ax2.add_feature(cfea.BORDERS, linestyle=':', edgecolor='darkgrey')
        ax2.add_feature(night.Nightshade(edata.attrs['time'], alpha=0.3))
        ax2.xaxis.set_major_formatter(lon_formatter)
        ax2.yaxis.set_major_formatter(lat_formatter)
        ax2.set_extent([-165.0, 45.0, 30.0, 80.0], crs=ccrs.PlateCarree())
        pdata['plotvar'] = eVar
        ax2 = gmdtools.plotFilledContours(ensmean, pdata, addTo=ax2)
        #plot all ensemble members as light grey quivers
        if doQuiver:
            for ense in collect:
                ax2 = gmdtools.plotVectors(ense, pdata, addTo=ax2, quiver=True, qstyle=qstyle)
            qstyle['alpha'] = 1
            qstyle['color'] = 'black'
        #then plot ensemble mean with streamlines...
        sstyle['color'] = 'darkblue'
        #collect Emag and calculate [mean/standard deviation] across ensemble members at each gridpoint
        ax2 = gmdtools.plotVectors(ensmean, pdata, addTo=ax2, quiver=False, maxVec=2.75,
                                   qstyle=qstyle, sstyle=sstyle)
        for aa in [ax, ax2]:
            aa.set_xticks([-150, -120, -90, -60, -30, 0, 30], crs=ccrs.PlateCarree())
            aa.set_yticks([35, 45, 55, 65, 75], crs=ccrs.PlateCarree())

        #now annotate panels with "Reference" and "Ensemble" (put text at Lisbon, Portugal)
        anchor = (-9.195, 38.744) #Lisbon #(-7.9304, 37.0194) #Faro, Port.
        #ensText = 'Ensemble Mean + $\sigma(|E|)$' if eVar=='Estd' else 'Ensemble Mean'
        ensText = 'Ensemble Mean'
        ax.text(anchor[0], anchor[1], 'Unperturbed', verticalalignment='top', weight='semibold',
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='None'), transform=ccrs.PlateCarree())
        ax2.text(anchor[0], anchor[1], ensText+'\n(N={0})'.format(len(collect)), 
                 verticalalignment='top', weight='semibold', color='darkblue',
                 bbox=dict(facecolor='white', alpha=0.3, edgecolor='None'), transform=ccrs.PlateCarree())

        #windows can't handle colons in filenames...
        isotime = edata.attrs['time'].isoformat()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, r'{0}_{1}.png'.format(pdata['plotvar'],
                                 isotime.replace(':',''))), dpi=300)
        plt.close()
    gmdtools.makeSymlinks(outdir, kind='E')


    #TODO: Make "hazard map" with dB/dt per pixel, per 20 minute averaging interval.
    #      Can use naive probabilistic classifier by taking 50th percentile from ensemble
    #      Can use peak or 95th percentile.
    #      Note that estimating the PDF using KDEs is similar in concept to
    #      Bayesian Model Averaging, but without bias-correction of the ensemble
    #      members.

    #TODO: Find dB/dt for each time, for each pixel, and display thresholded map of
    #      mean (median?)

    #TODO: Probably should use an equal-area projection for publishing results...
    #      Lambert Cylindrical?

if __name__=='__main__':
    #saveloc = 'E_maps_ens_mean'#'dBdt_indivMaps'
    saveloc = 'dBdt_indivMaps'
    saveloc = 'dBdt_AMR'
    if not os.path.isdir(saveloc):
        os.mkdir(saveloc)
    #singleRundBdt('20100405_MagG030', searchpatt='mag_*0833*[05]*', outdir=saveloc, links=False)
    singleRundBdt('AMRsouth', searchpatt='mag_*', outdir=saveloc, links=False)
    #singleRunE('20100405_QBC4_030', searchpatt='*083[01]*csv', outdir=saveloc, vecs=False, links=False)
    #singleRunE('20100405_Ensemble030', searchpatt='*083[01]*csv', outdir=saveloc, vecs=False, links=False)
    #singleRunEwithEnsemble('20100405_EnsembleOrig', '20100405_Ensemble*', searchpatt='geoe_*.csv', outdir=saveloc, eVar='Emag')
    #saveloc = 'E_maps_ens_std'#'dBdt_indivMaps'
    #if not os.path.isdir(saveloc):
    #    os.mkdir(saveloc)
    #singleRunEwithEnsemble('20100405_EnsembleOrig', '20100405_Ensemble*', searchpatt='geoe_*.csv', outdir=saveloc, eVar='Estd')
    #saveloc = 'E_maps_indiv'#'dBdt_indivMaps'
    #if not os.path.isdir(saveloc):
    #    os.mkdir(saveloc)
    #singleRunE('20100405_Ensemble001', searchpatt='geoe_*083245*.csv', outdir=saveloc, links=False)
    #singleRunE('20100405_Ensemble002', searchpatt='geoe_*083245*.csv', outdir=saveloc, links=False)
    #singleRunE('20100405_Ensemble042', searchpatt='geoe_*083245*.csv', outdir=saveloc, links=False)
