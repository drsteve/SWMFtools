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
        fig, ax = makeMap(maptime=mdata.attrs['time'], projection=pdata['plotprojection'])

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
        cmg = ax.pcolormesh(lons, mdata['Lat'], clipdata, transform=pdata['dataprojection'], cmap='seismic')
        skipPlot = True
    elif 'Estd' in plotvar:
        minplotcol = 2e-1
        maxplotcol = 3
        colmap = discrete_cmap(7, 'YlOrBr')
        clabel = '$\sigma$(|E|) [V/km]'
        clipdata = clipdata + 1e-6
        norm = mplc.LogNorm(vmin=3e-3, vmax=3)
    else:
        maxplotcol = 25
        colmap = 'magma_r'
        clabel = '(dB/dt)$_{H}$ [nT/s]'
        #cmg = ax.contourf(lons, mdata['Lat'], np.array(data), np.linspace(0,20,150),
        #                  transform=pdata['dataprojection'], cmap='magma_r',#'YlOrRd',
        #                  vmin=0, vmax=maxplotcol,
        #                  alpha=0.5, extend='max')
    if not skipPlot:    
        clipdata[clipdata>maxplotcol] = maxplotcol
        cmg = ax.contourf(lons, mdata['Lat'], clipdata, np.linspace(0,25,50),
                          transform=pdata['dataprojection'], cmap=colmap,#'YlOrRd',
                          vmin=minplotcol, vmax=maxplotcol, norm=norm,
                          alpha=0.5)#, extend='max')
        cmg.cmap.set_over('k')
    cmg.set_clim(minplotcol, maxplotcol)
    #cmg.cmap.set_under(cmg.cmap(0))
    map_dummy = plt.cm.ScalarMappable(cmap=colmap, norm=norm)
    map_dummy.set_array(np.array(data))
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


def plotVectors(mdata, pdata, addTo=None, quiver=False, maxVec=2.5, qstyle=None, sstyle=None):
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
    plotvar = pdata['plotvec']
    if 'Lat_raw' in mdata:
        if quiver:
            lons = mdata['Lon_raw'][::2]
            lats = mdata['Lat_raw'][::2]
            vec_e = mdata['{0}e_raw'.format(plotvar)][::2]
            vec_n = mdata['{0}n_raw'.format(plotvar)][::2]
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
        ax.set_title('{0}[{1}]\n{2}'.format(plotvar, 
                                            mdata[plotvar].attrs['units'], 
                                            mdata.attrs['time'].isoformat()))
    else:
        ax = addTo
    if quiver:
        if qstyle is None:
            qstyle = dict()
        ax.quiver(lons, lats, vec_e, vec_n, transform=pdata['dataprojection'], zorder=33, **qstyle)
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


def makeMap(maptime=dt.datetime.now(), figsize=(10,4), projection=ccrs.PlateCarree(),
            extent=[-165.0, 145.0, 30.0, 80.0], nightshade=True):
    #PlateCarree is an equirectangular projection where lines of equal longitude
    #are vertical and lines of equal latitude are horizontal. Spacing between lats
    #and longs are also equal.
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=projection)
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


def northAmerica(fname):
    import matplotlib.ticker as mticker
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    #read
    data = readEcsv(fname)
    #setup
    myproj = ccrs.Mercator()
    pdata = {}
    pdata['plotprojection']=myproj
    pdata['plotvar']='Emag'
    pdata['dataprojection']=ccrs.PlateCarree()
    pdata['maxplotcol']=5
    pdata['plotvec']='E'
    #plot
    fig, ax = makeMap(maptime=data.attrs['time'], projection=myproj, extent=[-140,-35,25,70])
    plotFilledContours(data,pdata,addTo=ax)
    plotVectors(data,pdata,addTo=ax)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.6, linestyle='--')
    gl.xlocator = mticker.FixedLocator([-140, -120, -100, -80, -60, -40, -20, 0])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabels_top = False
    gl.ylabels_right = False
    ax.set_title(data.attrs['time'].isoformat())
    #plt.show()
