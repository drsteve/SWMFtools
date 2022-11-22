import copy
import glob
import re
import os
import itertools
import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import cartopy.crs as ccrs
import cartopy.feature as cfea
import cartopy.feature.nightshade as night
import cartopy.util as cuti
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import spacepy.datamodel as dm
import spacepy.toolbox as tb


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def plotFilledContours(mdata, pdata, addTo=None):
    plotvar = pdata['plotvar']
    try:
        data, lons = cuti.add_cyclic_point(np.transpose(mdata[plotvar]), 
                                           mdata['Lon'])
    except ValueError:
        try:
            data, lons = cuti.add_cyclic_point(mdata[plotvar], mdata['Lon'])
        except:
            data = mdata[plotvar]
            lons = mdata['Lon']
    if addTo is None:
        fig, ax = makeMap(maptime=mdata.attrs['time'], projection=pdata['plotprojection'],
                          extent=[-145.0, 15.0, 30.0, 80.0])
        stn_list = ['FRD', 'OTT', 'PBQ']
        lon_list = [-77.3729, -75.552, -77.745]
        lat_list = [38.2047, 45.403, 55.277]
        ax.plot(lon_list, lat_list, color='k', linestyle='none', marker='o',
                markersize=2, transform=ccrs.PlateCarree())
        for idx, stn in enumerate(stn_list):
            ax.text(lon_list[idx] + 2, lat_list[idx] - 2.5, stn,
                    horizontalalignment='left', fontsize='small',
                    transform=ccrs.PlateCarree())
        if pdata['shapes'] is not None:
            ax.add_geometries(pdata['shapes'], pdata['plotprojection'], 
                              edgecolor='black', facecolor='none')
        #ax.gridlines()
        ax.set_xticks([-150, -120, -90, -60, -30, 0, 30], crs=ccrs.PlateCarree())
        ax.set_yticks([35, 45, 55, 65, 75], crs=ccrs.PlateCarree())
        ax.set_title('{0}'.format(mdata.attrs['time'].isoformat()))
    else:
        ax = addTo
    norm = mplc.Normalize()
    clipdata = np.array(data)
    minplotcol = 0.0
    skipPlot = False
    if 'Emag' in plotvar:
        maxplotcol = 2.5 if 'maxplotcol' not in pdata else pdata['maxplotcol']
        norm = mplc.Normalize(vmin=minplotcol, vmax=pdata['maxplotcol'])
        colmap = 'magma_r'
        clabel = '|E| [V/km]'
    elif 'jr' in plotvar:
        minplotcol = -1.
        maxplotcol = 1.
        colmap = 'seismic'
        clabel = 'j$_{R}$ [mA/m$^{2}$]'
        #cmg = ax.contourf(lons, mdata['Lat'], clipdata, np.linspace(-2,2,50),
        #              transform=pdata['dataprojection'], cmap=colmap,#'YlOrRd',
        #              vmin=minplotcol, vmax=maxplotcol, norm=norm,
        #              alpha=0.5)#, extend='max')
        cmg = ax.pcolormesh(lons, mdata['Lat'], clipdata,
                            transform=pdata['dataprojection'],
                            cmap='seismic')
        skipPlot = True
    elif 'Estd' in plotvar:
        minplotcol = 2e-1
        maxplotcol = 3
        colmap = discrete_cmap(7, 'YlOrBr')
        clabel = '$\sigma$(|E|) [V/km]'
        clipdata = clipdata + 1e-6
        norm = mplc.LogNorm(vmin=3e-3, vmax=3)
    else:
        maxplotcol = 10#25
        colmap = 'magma_r'
        clabel = '(dB/dt)$_{H}$ [nT/s]'
        #cmg = ax.contourf(lons, mdata['Lat'], np.array(data), np.linspace(0,20,150),
        #                  transform=pdata['dataprojection'], cmap='magma_r',#'YlOrRd',
        #                  vmin=0, vmax=maxplotcol,
        #                  alpha=0.5, extend='max')
    if not skipPlot:    
        clipdata[clipdata>maxplotcol] = maxplotcol
        cmap = copy.copy(plt.get_cmap(colmap))
        cmg = ax.contourf(lons, mdata['Lat'], clipdata, np.linspace(0,25,maxplotcol*2),
                          transform=pdata['dataprojection'], cmap=cmap,#'YlOrRd',
                          vmin=minplotcol, vmax=maxplotcol, norm=norm,
                          alpha=0.5)#, extend='max')

        cmg.cmap.set_over('k')
    cmg.set_clim(minplotcol, maxplotcol)
    #cmg.cmap.set_under(cmg.cmap(0))
    map_dummy = plt.cm.ScalarMappable(cmap=colmap, norm=norm)
    map_dummy.set_array(np.array(clipdata))
    map_dummy.set_clim(minplotcol, maxplotcol)
    cax = plt.colorbar(map_dummy, ax=ax, extend='max', shrink=0.4, pad=0.04, drawedges=False)
    cax.set_label(clabel)
    return ax


def plotContour(mdata, pdata, addTo=None):
    plotvar = pdata['plotvar']
    data, lons = cuti.add_cyclic_point(np.transpose(mdata[plotvar]), 
                                       mdata['Lon'])
    if addTo is None:
        fig, ax = makeMap(maptime=mdata.attrs['time'], projection=pdata['plotprojection'])

        if pdata['shapes'] is not None:
            ax.add_geometries(pdata['shapes'], pdata['plotprojection'], 
                              edgecolor='black', facecolor='none')
        #ax.gridlines()
        ax.set_xticks([-150, -120, -90, -60, -30, 0, 30], crs=ccrs.PlateCarree())
        ax.set_yticks([35, 45, 55, 65, 75], crs=ccrs.PlateCarree())
        ax.set_title('{0}[{1}]\n{2}'.format(plotvar, 
                                            mdata[plotvar].attrs['units'], 
                                            mdata.attrs['time'].isoformat()))
    else:
        ax = addTo
    cax = ax.contourf(lons, mdata['Lat'], np.array(data), [pdata['thresh'], data.max()], 
                      transform=pdata['dataprojection'], colors=pdata['ccolor'],
                      alpha=0.75)
    return ax


def plotVectors(mdata, pdata, addTo=None, quiver=False, maxVec=2.5,
                qstyle=None, sstyle=None):
    '''

    Optional arguments
    ------------------
    addTo : Axes or None
        If None (default), then a new figure is created. Otherwise the supplied
        Axes are used to plot the vectors on. The axes are assumed to support
        projections (e.g. cartopy)
    quiver : boolean
        If True makes a quiver plot. If False (default) draws streamlines scaled
        in width by the vector magnitude.
    maxVec : float
        Limit for the linewidth of streamlines.
    qstyle : dict or None
        Dictionary containing plot style kwargs for quiver plots
    '''
    from matplotlib.colors import LogNorm
    plotvar = pdata['plotvec']
    if 'Lat_raw' in mdata:
        if quiver:
            lons = mdata['Lon_raw'][::1]
            lats = mdata['Lat_raw'][::1]
            vec_e = mdata['{0}e_raw'.format(plotvar)][::1]
            vec_n = mdata['{0}n_raw'.format(plotvar)][::1]
        else:
            lons = mdata['Lon_raw']
            lats = mdata['Lat_raw']
            vec_e = mdata['{0}e_raw'.format(plotvar)]
            vec_n = mdata['{0}n_raw'.format(plotvar)]
    else:
        latlon = list(itertools.product(mdata['Lon'], mdata['Lat']))
        lons = np.array(latlon)[:,0]
        lats = np.array(latlon)[:,1]
        vec_e = np.ravel(mdata['{0}e'.format(plotvar)])
        vec_n = np.ravel(mdata['{0}n'.format(plotvar)])
    if addTo is None:
        fig, ax = makeMap(maptime=mdata.attrs['time'], projection=pdata['plotprojection'])
        ax.set_xticks([-150, -120, -90, -60, -30, 0, 30], crs=ccrs.PlateCarree())
        ax.set_yticks([35, 45, 55, 65, 75], crs=ccrs.PlateCarree())
        ax.set_title('{0} [{1}]\n{2}'.format(plotvar, 
                                            'V/km', 
                                            mdata.attrs['time'].isoformat()))
    else:
        ax = addTo
    if quiver:
        def logscl(u, v):
            arr_len = np.sqrt(u*u + v*v)
            len_adj = np.log10(arr_len + 1)/arr_len
            return u*len_adj, v*len_adj

        if qstyle is None:
            qstyle = dict()
            qstyle['width'] = 0.004
            qstyle['headlength'] = 4
            qstyle['headaxislength'] = 4
            qstyle['cmap'] = 'plasma' #'inferno'
        mynorm = LogNorm(vmin=1e-2, vmax=5)
        vec_mag = np.linalg.norm(np.c_[vec_e, vec_n], axis=-1)
        pltu, pltv = logscl(vec_e, vec_n)
        qp = ax.quiver(lons, lats, pltu, pltv, vec_mag,
                       transform=pdata['dataprojection'], zorder=33,
                       norm=mynorm, **qstyle)
        cmap = qp.get_cmap()
        l0 = 5
        l1 = 1
        l2 = 0.5
        l3 = 0.1
        lab0magcol = cmap(mynorm(l0))
        lab1magcol = cmap(mynorm(l1))
        lab2magcol = cmap(mynorm(l2))
        lab3magcol = cmap(mynorm(l3))
        ax.quiverkey(qp, 0.58, 0.05, np.log10(l0+1), '{} V/km'.format(l0), labelpos='E',
                     transform=pdata['dataprojection'], color=lab0magcol, coordinates='figure')
        ax.quiverkey(qp, 0.43, 0.05, np.log10(l1+1), '{} V/km'.format(l1), labelpos='E',
                     transform=pdata['dataprojection'], color=lab1magcol, coordinates='figure')
        ax.quiverkey(qp, 0.27, 0.05, np.log10(l2+1), '{} V/km'.format(l2), labelpos='E',
                     transform=pdata['dataprojection'], color=lab2magcol, coordinates='figure')
        ax.quiverkey(qp, 0.12, 0.05, np.log10(l3+1), '{} V/km'.format(l3), labelpos='E',
                     transform=pdata['dataprojection'], color=lab2magcol, coordinates='figure')
        plt.colorbar(qp, label='|E$_H$| [V/km]', extend='both')
    else:
        if sstyle is None:
            sstyle = dict()
        if 'arrowsize' not in sstyle: sstyle['arrowsize'] = 0.75
        if 'color' not in sstyle: sstyle['color'] = 'black'
        if 'linewidth' not in sstyle:
            mag = np.sqrt(vec_e**2+vec_n**2)
            lw = ((2.5*mag)+0.5)/(2.5*maxVec)
            lw[lw>=maxVec] = (2.5*maxVec)+0.5
            sstyle['linewidth'] = lw
        ax.streamplot(lons, lats, vec_e, vec_n, transform=pdata['dataprojection'], zorder=33, **sstyle)
    return ax


def makeMap(figure=None, axes=None, maptime=dt.datetime.now(), figsize=(10,4), projection=ccrs.PlateCarree(),
            extent=[-165.0, 145.0, 30.0, 80.0], nightshade=True):
    #PlateCarree is an equirectangular projection where lines of equal longitude
    #are vertical and lines of equal latitude are horizontal. Spacing between lats
    #and longs are also equal.
    if figure is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = figure
    if axes is None:
        ax = plt.axes(projection=projection)
    else:
        ax = axes
    ax.coastlines(color='darkgrey')
    ax.add_feature(cfea.BORDERS, linestyle=':', edgecolor='darkgrey')
    if nightshade:
        ax.add_feature(night.Nightshade(maptime, alpha=0.3))
    lon_formatter = LongitudeFormatter(number_format='.1f',
                                       degree_symbol='',
                                       dateline_direction_label=True)
    lat_formatter = LatitudeFormatter(number_format='.1f',
                                      degree_symbol='')
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    return fig, ax


def forwardDiff(a, b, c, dt):
    return (-c + 4*b - 3*a)/(2*dt)

def backwardsDiff(x, y, z, dt):
    return (3*z - 4*y + x)/(2*dt)

def centralDiff(a, c, dt):
    return (c-a)/(2*dt)


class ElecData(dm.SpaceData):
    def addTimes(self, filenames):
        toAdd = [readEcsv(fn) for fn in filenames]
        if 'times' not in self.attrs:
            self.attrs['times'] = [self.attrs['time']]
        for ta in toAdd:
            self.attrs['times'].append(ta.attrs['time'])
        for key in self:
            if 'Lat' in key or 'Lon' in key: continue
            if '_raw' in key:
                self[key] = np.stack([self[key], *[ta[key] for ta in toAdd]])
                self[key] = self[key].T
            else:
                self[key] = np.dstack([self[key], *[ta[key] for ta in toAdd]])


def readEcsv(fname):
    try:
        import pandas as pd
        has_pandas = True
    except (ImportError, ModuleNotFoundError):
        has_pandas = False

    if has_pandas:
        dframe = pd.read_csv(fname, sep=',')
        rawdata = dframe.values  # get underlying numpy array
    else:
        rawdata = np.loadtxt(fname, delimiter=',', skiprows=1)

    edata = ElecData()
    #fileidx = int(re.search('\d{4}', fname).group())
    #offset = fileidx*5
    tstr = re.search('\d{8}-(\d{6})', fname).group()
    edata.attrs['time'] = dt.datetime(int(tstr[:4]), int(tstr[4:6]), int(tstr[6:8]), int(tstr[9:11]), int(tstr[11:13]), int(tstr[13:15]))
    edata['Lat_raw'] = rawdata[:, 0]
    edata['Lon_raw'] = rawdata[:, 1]
    edata['Ee_raw'] = rawdata[:, 2]
    edata['En_raw'] = rawdata[:, 3]
    nlat = len(set(edata['Lat_raw']))
    nlon = len(set(edata['Lon_raw']))
    edata['Lat'] = np.reshape(rawdata[:, 0], [nlat, nlon])[:,0]
    edata['Lon'] = np.reshape(rawdata[:, 1], [nlat, nlon])[0,:]
    edata['Ee'] = np.reshape(rawdata[:, 2], [nlat, nlon])
    edata['En'] = np.reshape(rawdata[:, 3], [nlat, nlon])
    edata['Emag'] = np.sqrt(edata['Ee']**2.0 + edata['En']**2.0)

    return edata


def makeSymlinks(dirname, kind='dbdt'):
    #make symlinks to images with numeric naming so it can be easily passed to ffmpeg
    pngfns = sorted(glob.glob(os.path.join(dirname,'{0}*png'.format(kind))))
    for idx, fn in enumerate(pngfns):
        os.symlink(os.path.abspath(fn), 
                   os.path.abspath(os.path.join(dirname,
                   'img{0:04d}.png'.format(idx))))


def northAmerica(fname, emin=1e-2, emax=5, conus=False, quiver=False):
    import matplotlib.ticker as mticker
    from matplotlib import gridspec
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import cartopy.io.shapereader as shpreader
    #read
    data = readEcsv(fname)
    #setup
    myproj = ccrs.Mercator()
    pdata = {}
    pdata['plotprojection'] = myproj
    pdata['plotvar'] = 'Emag'
    pdata['dataprojection'] = ccrs.PlateCarree()
    pdata['maxplotcol'] = emax
    pdata['plotvec'] = 'E'
    #plot
    extent = [-130, -60, 25, 50] if conus else [-140, -35, 25, 65]
    labels = list(range(extent[0], extent[1], 20))
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(3, 3)
    ax = fig.add_subplot(gs[:, :], projection=myproj)
    ax2 = fig.add_subplot(gs[2, :])
    fig, ax = makeMap(figure=fig, axes=ax, figsize=(10,4), maptime=data.attrs['time'],
                      projection=myproj, extent=extent, nightshade=False)
    states = list(shpreader.Reader('/home/smorley/shape/cb_2018_us_state_500k.shp').geometries())
    ax.add_geometries(states, pdata['dataprojection'], edgecolor='silver', facecolor='none')
    #plotFilledContours(data,pdata,addTo=ax)
    if not quiver:
        from scipy.interpolate import interp2d
        from matplotlib.colors import LogNorm
        extentin0360 = (np.asarray(extent) + 360) % 360
        ifun = interp2d(data['Lon'], data['Lat'], data['Emag'], kind='cubic')
        lons = np.arange(extentin0360[0], extentin0360[1], 0.1)
        lats = np.arange(extentin0360[2], extentin0360[3], 0.1)
        emag = ifun(lons, lats)
        emin = emin if emin else 1e-3
        emag[emag < emin] = emin
        cm = ax.pcolormesh(lons, lats, emag,
                           #vmin=0, vmax=emax,
                           norm=LogNorm(vmin=emin, vmax=emax),
                           shading='nearest',
                           transform=ccrs.PlateCarree(),
                           cmap='inferno')
        fig.colorbar(cm, ax=ax, extend='both', label='|E$_{H}$| [V/km]')
    plotVectors(data, pdata, addTo=ax, quiver=quiver)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.6, linestyle='--')
    gl.xlocator = mticker.FixedLocator(labels)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(data.attrs['time'].isoformat())


def world(fname, emax=5):
    import matplotlib.ticker as mticker
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    #read
    data = readEcsv(fname)
    #setup
    myproj = ccrs.Mercator()
    pdata = {}
    pdata['plotprojection'] = myproj
    pdata['plotvar'] = 'Emag'
    pdata['dataprojection'] = ccrs.PlateCarree()
    pdata['maxplotcol'] = emax
    pdata['plotvec'] = 'E'
    #plot
    fig, ax = makeMap(maptime=data.attrs['time'], projection=myproj, extent=[-175,175,-73,73])
    plotFilledContours(data, pdata, addTo=ax)
    plotVectors(data, pdata, addTo=ax)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.6, linestyle='--')
    gl.xlocator = mticker.FixedLocator(list(range(-160, 180, 40)))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(data.attrs['time'].isoformat())
    #plt.show()


if __name__ == '__main__':
    import glob
    import os
    runname = 'scaledA2_3d' # '20041108_3d' # 'scaledA2_3d'
    hdir = os.path.expanduser('~')
    fnames = sorted(glob.glob(hdir + '/tmp/*0435*.csv'))

    def fmtTime(estr, timeonly=True):
        yr = int(estr[1:5])
        mn = int(estr[5:7])
        dy = int(estr[7:9])
        hr = int(estr[10:12])
        mi = int(estr[12:14])
        sc = int(estr[14:])
        if timeonly:
            tmstr = ''
        else:
            tmstr = '{}-{:02d}-{:02d}T'.format(yr, mn, dy)
        tmstr += '{:02d}:{:02d}:{:02d}'.format(hr, mi, sc)
        return tmstr

    print('Processing {} input files'.format(len(fnames)))
    for fname in fnames:
        indir, infname = os.path.split(fname)
        outdir = os.path.split(indir)[1]
        outfn1 = infname.split('_')[-1].split('.')[0]
        outfname_q = os.path.join(outdir, outfn1+'_quiv_IEEE.png')
        #outfname_m = os.path.join(outdir, outfn1+'_mesh.png')
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        #northAmerica(fname, emax=5, conus=True)
        #plt.title('{}; {}'.format(runname, fmtTime(outfn1)))
        #plt.savefig(outfname_m)
        #plt.close('all')
        northAmerica(fname, emax=5, conus=True, quiver=True)

        #plt.title('{}; {}'.format(runname, fmtTime(outfn1)))
        plt.title('Scenario A2; {}'.format(runname, fmtTime(outfn1)))
        plt.show()
        #plt.savefig(outfname_q)
        #plt.close('all')
