import datetime as dt
import numpy as np
from scipy import linalg
from scipy.signal import decimate
import matplotlib.pyplot as plt
import spacepy.toolbox as tb
import spacepy.plot as splot
import spacepy.pybats.bats
import supermag_parser
splot.style('default')
try:
    assert origdata
except:
    origdata = spacepy.pybats.bats.MagFile('magnetometers_e20000812-010000.mag')
    origdata.calc_h()
    origdata.calc_dbdt()

smdata = supermag_parser.supermag_parser('supermag-20000810-20000812.txt')

stations = ['YKC', 'MEA', 'NEW', 'FRN', 'IQA', 'PBQ', 'OTT', 'FRD', 'VIC']
stations = [key for key in origdata.keys() if len(key)==3]

for stat in stations:
    if stat not in smdata.station: continue
    subset = origdata[stat]
    simtime = subset['time'][::6]
    dBdte = decimate(subset['dBdte'], 6)
    dBdtn = decimate(subset['dBdtn'], 6)
    dBdth = np.array([linalg.norm([dBdtn[i], dBdte[i]]) for i in range(len(dBdtn))])
    smstat = smdata.station[stat]
    Bdoth = np.array([linalg.norm(smstat['Bdot'][i,:2]) for i in range(len(smstat['Bdot']))])
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)
    ax.plot(simtime, dBdth, 'b-', alpha=0.4)
    ax.plot(smstat['time'], Bdoth, 'r-', alpha=0.4)
    run20, t20 = tb.windowMean(dBdth, time=simtime, winsize=dt.timedelta(minutes=20), overlap=dt.timedelta(0), st_time=dt.datetime(2000,8,12,1), op=np.max)
    ax.plot(t20, run20, marker='o', color='b', linestyle='none', markersize=3, label='SWMF')
    obs20, t20 = tb.windowMean(Bdoth, time=smstat['time'], winsize=dt.timedelta(minutes=20), overlap=dt.timedelta(0), st_time=dt.datetime(2000,8,12,1), op=np.max)
    ax.plot(t20, obs20, marker='x', color='r', linestyle='none', markersize=3, label='Obs')
    ax.set_ylabel('1-min dB/dt$_{H}$ [nT/s]')
    ax.set_xlabel('2000-08-12')
    splot.applySmartTimeTicks(ax, subset['time'], dolimit=True)
    plt.legend()
    plt.title(stat)
    plt.tight_layout()
    #plt.show()
    plt.savefig('Aug2000_dBdth_{}.png'.format(stat))
    plt.close()
