import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import spacepy.plot as splot

splot.revert_style()

def read_log(fname):
     with open(fname) as fh:
         lines = fh.readlines()
     head = lines[2][2:]
     nlines = int(lines[1].strip().split()[-2])
     data = np.zeros([nlines, 12]).astype(object)
     for idx, line in enumerate(lines[3:]):
         use = line.strip().split()
         use[1] = dt.datetime.strptime(use[1], '%Y%m%d-%H%M%S')
         for ii, val in enumerate(use):
             if ii == 1:
                 data[idx, ii] = val
             else:
                 data[idx, ii] = float(val)
     return data

if __name__=='__main__':
    datA2 = read_log('blake_scaledA2_3d.log')
    fig = plt.figure(figsize=(9,4))
    ax = fig.add_subplot(111)
    ax.plot(datA2[30:-30,1], datA2[30:-30,-2], 'r-', alpha=0.9, label='max in RTS-GMLC-GIC-EAST convex hull')
    ax.plot(datA2[30:-30,1], datA2[30:-30,-1], 'k-', alpha=0.5, label='mean in RTS-GMLC-GIC-EAST convex hull')
    splot.applySmartTimeTicks(ax, datA2[30:-30,1])
    ax.axvline(dt.datetime(2004, 11, 9, 4, 35), linestyle='--', color='k')
    ax.set_ylabel('|E$_{H}$| [V/km]')
    ax.set_ylim([0, 5])
    ax.legend()

    plt.savefig('rts-gmlc-time_series.png')