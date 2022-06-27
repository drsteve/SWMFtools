import os
import glob
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import spacepy.time as spt
import spacepy.toolbox as tb
import spacepy.datamodel as dm
from spacepy import pybats
import missing  # taken from advect1d

def loadIMFdata(evnum, fform='IMF_rc{:03d}_*.dat', datadir='.'):
    # select file given event number
    datapath = os.path.abspath(datadir)
    gterm = os.path.join(datapath, fform.format(evnum))
    fnl = glob.glob(gterm)[0]
    # load datafile and make time relative to start
    imfdata = pybats.ImfInput(fnl)
    imfdata['time'] = imfdata['time'] - imfdata['time'][0]
    return imfdata

# TODO
# add argument parser
# Take event numbers as args and stitch together in order
if __name__ == '__main__':
    events = [57, 103, 156, 259, 443]  # original chrono order
    events = [57, 78, 103, 156, 259]  # slow SW substituted, chrono
    events = [1, 57, 103, 156, 259]
    #events = [152, 253, 258, 266, 295]  # original chrono order NEK
    #events = [443, 156, 259, 57, 103]  # initial random select
    # If we want a random ordering, uncomment next line
    #events = np.random.choice(events, len(events), replace=False)


    datalist = []
    for ev in events:
        datalist.append(loadIMFdata(ev))

    # now loop over events and adjust times
    starttime = dt.datetime(1999, 6, 1)
    f107 = 170  # not using here, but noted for use in PARAM file(s)
    for data in datalist:
        lsttime = data['time'][-1]
        hr6 = dt.timedelta(hours=12)
        trail6  = tb.tOverlapHalf([lsttime-hr6, lsttime], data['time'])
        if data['ux'][trail6[0]] < data['ux'][-1]:
            for key in data:
                data[key] = data[key][:trail6[0]]
        data['time'] = data['time'] + starttime
        starttime = data['time'][-1] + dt.timedelta(hours=3)

    # append in order
    imfdata = datalist[0]
    keylist = list(imfdata.keys())
    for data in datalist[1:]:
        for key in keylist:
            imfdata[key] = dm.dmarray.concatenate(imfdata[key], data[key])

    # find gaps and fill with badval, then do noisy gap filling
    keylist.pop(keylist.index('time'))
    keylist.pop(keylist.index('pram'))
    newtime = spt.tickrange(imfdata['time'][0], imfdata['time'][-1], dt.timedelta(minutes=1))
    innew, inold = tb.tCommon(newtime.UTC, imfdata['time'], mask_only=True)
    synthdata = pybats.ImfInput(filename=False, load=False, npoints=len(newtime))
    synthdata['time'] = newtime.UTC
    for key in keylist:
        synthdata[key] = dm.dmfilled(len(newtime), fillval=-999)
        synthdata[key][innew] = imfdata[key][inold]
        cval = True if key == 'rho' else False
        # missing.fill_gaps currently adds noise to linear interp.
        # possibly replace with "sigmoid_like" interpolation
        # see reddit.com/r/gamedev/comments/4xkx71/sigmoidlike_interpolation/
        synthdata[key] = missing.fill_gaps(synthdata[key], fillval=-999, noise=True, constrain=cval, method='sigmoid')
    synthdata.quicklook()
    plt.savefig('IMF_stitched_{:03d}_{:03d}_{:03d}_{:03d}_{:03d}.png'.format(*events))
    synthdata.write('IMF_stitched_{:03d}_{:03d}_{:03d}_{:03d}_{:03d}.dat'.format(*events))
