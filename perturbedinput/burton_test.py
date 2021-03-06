import os
import glob
import numpy as np
import scipy.optimize as opt
import matplotlib as mpl
import matplotlib.pyplot as plt
import spacepy.pybats as bats
import spacepy.pybats.kyoto as kyo
import spacepy.empiricals as emp
import spacepy.toolbox as tb
import spacepy.time as spt
import spacepy.plot as splot
splot.style('spacepy')
mpl.rcParams.update({'font.size': 15})


def Dst_Burton(initDst, v, Bz, dt=1, alpha=-4.5, tau=7.7):
    '''Advances by timestep dt [hours], adds difference to Dst*'''
    Bsouth = Bz.copy()
    Bsouth[Bz >= 0] = 0  # half-wave rectified
    E = 1e-3 * v * Bsouth
    c1 = alpha*E
    currentDst = initDst
    newDst = np.empty_like(Bz)
    for idx, val in enumerate(c1):
        c2 = currentDst/tau
        deltaDst = (val - c2)*dt
        if not np.isnan(deltaDst):
            currentDst += deltaDst
        newDst[idx] = currentDst
    return newDst


def Dst_OBrien(initDst, v, Bz, dt=1, alpha=-4.4):
    '''Using O'Brien & McPherron formulation.
    doi: 10.1029/1998JA000437
    '''
    Ecrit = 0.49
    Bsouth = Bz.copy()
    Bsouth[Bz >= 0] = 0  # half-wave rectified
    E = 1e-3 * v * Bsouth

    def getQ(alpha, E):  # nT/h
        mask = E <= Ecrit
        Q = alpha*(E-Ecrit)
        Q[mask] = 0
        return Q

    def getTau(E, dt=dt):
        return 2.4*np.exp(9.74/(4.69+E))
    c1 = getQ(alpha, E)
    currentDst = initDst
    newDst = np.empty_like(Bz)
    for idx, val in enumerate(c1):
        c2 = currentDst/getTau(E[idx])
        deltaDst = (val - c2)*dt
        if not np.isnan(deltaDst):
            currentDst += deltaDst
        newDst[idx] = currentDst
    return newDst


def inverseBurton(dst, dt=1, alpha=-4.5, tau=7.7):
    """Given a sequence of Dst, get the rectified vBz

    Example
    =======
    import numpy as np
    import matplotlib.pyplot as plt
    v = 400*np.ones(24)
    bz = 5*np.sin(np.arange(24))
    vbz = v*bz
    dst = Dst_Burton(0, v, bz)
    recon = inverseBurton(dst)
    plt.plot(recon, 'r-', label='Reconstructed')
    plt.plot(vbz, 'k-', label='Original')
    plt.legend()
    plt.show()
    """
    vbz = np.zeros(dst.shape)
    c2 = dst/tau
    deltaDst = dst[1:] - dst[:-1]
    lhs = (deltaDst/dt) + c2[:-1]
    E = lhs/alpha
    vbz[1:] = E/1e-3
    return vbz


def inverseOBrienMcPherron(dst, dt=1, alpha=-4.4):
    """
    """
    vbz = np.zeros(dst.shape)

    def penFunc(E, dst_prev, dst_curr, dt):
        Ecrit = 0.49
        delta_dst = dst_curr - dst_prev
        tau = 2.4*np.exp(9.74 / (4.69 + E))
        guess = alpha*(E - Ecrit) - (dst_curr / tau)
        penalty = np.abs(guess - delta_dst/dt)
        return penalty

    for idx, dst_prev, dst_curr in zip(range(len(dst)-1), dst[:-1], dst[1:]):
        # Optimize using bounded Brent's method to find vbz
        res = opt.minimize_scalar(penFunc, args=(dst_prev, dst_curr, dt), bracket=(0, 1e5),
                                  bounds=(0, 1e7), method='bounded', tol=None, options=None)
        E = res.x
        vbz[idx+1] = E/1e-3
    return vbz


def invert_example():
    """Test case for reverse engineering rough solar wind params

    We look here at the 1989 storm as studied by Nagatsuma (2015)
    and Boteler (2019)
    """
    import datetime as dt
    import spacepy.toolbox as tb
    # fetch doesn't respect the days (it grabs 1 month at a time)
    # but we'll give the days we're inerested in for reference
    st_tup = (1989, 3, 12)
    en_tup = (1989, 3, 15)
    ky_dat = kyo.fetch('dst', st_tup, en_tup)
    st = dt.datetime(*st_tup)
    en = dt.datetime(*en_tup)
    inds = tb.tOverlapHalf([st, en], ky_dat['time'])
    vbz = inverseBurton(ky_dat['dst'][inds])
    vbz_OB = inverseOBrienMcPherron(ky_dat['dst'][inds])

    fig, ax = plt.subplots(2, sharex=True, figsize=(10, 4.5))
    ax[0].plot(ky_dat['time'][inds], -vbz, 'r-', label='Burton')
    ax[0].set_ylabel('v.B$_{south}$ [nT.(km/s)$^{2}$]')
    ax[0].plot(ky_dat['time'][inds], -vbz_OB, 'b-', label='O-M')
    ax[0].set_ylabel('v.B$_{south}$ [nT.(km/s)$^{2}$]')
    ax[0].legend()
    ax[1].plot(ky_dat['time'][inds], ky_dat['dst'][inds])
    ax[1].set_ylabel('Dst [nT]')
    plt.show()


def main(infiles=None):
    # read SWMF ImfInput file
    # Originally written to examine data from simulations
    # by Morley, Welling and Woodroffe (2018). See data at
    # Zenodo (https://doi.org/10.5281/zenodo.1324562)
    infilename = 'Event5Ensembles/run_orig/IMF.dat'
    eventIMF = bats.ImfInput(filename=infilename)
    data1 = bats.LogFile('Event5Ensembles/run_orig/GM/IO2/log_e20100404-190000.log')

    # #read IMF for 10 events...
    # infiles = ['Event5Ensembles/run_{:03d}/IMF.dat'.format(n)
    #            for n in [32,4,36,10,13,17,18,20,24,29]]
    # read IMF files for all ensemble members
    if infiles is None:
        infiles = glob.glob('Event5Ensembles/run_???/IMF.dat')
        subsetlabel = False
    else:
        # got list of run directories from Dst/Kp plotter
        subsetlabel = True
        infiles = [os.path.join(d, 'IMF.dat') for d in infiles]
        nsubset = len(infiles)
    eventlist = [bats.ImfInput(filename=inf) for inf in infiles]

    tstart = eventIMF['time'][0]
    tstop = eventIMF['time'][-1]
    sym = kyo.KyotoSym(lines=bats.kyoto.symfetch(tstart, tstop))
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    for ev in eventlist:
        gco = '{}'.format(np.random.randint(5, 50)/100.0)
        pred01B = Dst_Burton(sym['sym-h'][0], ev['ux'], ev['bz'], dt=1./60)
        pred01O = Dst_OBrien(sym['sym-h'][0], ev['ux'], ev['bz'], dt=1./60)
        ax1.plot(ev['time'], pred01B, c=gco, alpha=0.5)
        ax2.plot(ev['time'], pred01O, c=gco, alpha=0.5)
    # ax1.plot(sym['time'], sym['sym-h'], lw=1.5, c='crimson', label='Sym-H')
    evtime = spt.Ticktock(eventIMF['time']).RDT
    datime = spt.Ticktock(data1['time']).RDT
    simDst = tb.interpol(evtime, datime, data1['dst'])
    # ax1.plot(eventIMF['time'], simDst+11-(7.26*eventIMF['pram']), lw=1.5, c='seagreen', label='Sym-H (Press.Corr.)')
    # ax2.plot(sym['time'], sym['sym-h'], lw=1.5, c='crimson')
    # ax2.plot(eventIMF['time'], simDst+11-(7.26*eventIMF['pram']), lw=1.5, c='seagreen', label='Sym-H (Press.Corr.)')
    ax1.plot(data1['time'], data1['dst'], linewidth=1.5, color='crimson', alpha=0.65, label='SWMF')
    ax2.plot(data1['time'], data1['dst'], linewidth=1.5, color='crimson', alpha=0.65)
    ax1.legend()
    splot.applySmartTimeTicks(ax1, [tstart, tstop])
    splot.applySmartTimeTicks(ax2, [tstart, tstop], dolabel=True)
    ax1.set_ylabel('Sym-H [nT]')
    ax2.set_ylabel('Sym-H [nT]')
    ax1.text(0.05, 0.05, "Burton et al.", transform=ax1.transAxes)
    ax2.text(0.05, 0.05, "O'Brien et al.", transform=ax2.transAxes)
