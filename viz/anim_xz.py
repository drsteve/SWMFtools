import glob

import matplotlib.pyplot as plt
from spacepy.pybats import bats

import celluloid




def makeplot(fname, ax):
    data = bats.Bats2d(fn)
    data.add_b_magsphere(target=ax, nOpen=10, nClosed=8)
    ti_str = '{0}'.format(data.attrs['time'])
    ax.set_xlim([-32, 16])
    ax.set_ylim([-24, 24])
    ax.set_xlabel('X$_{GSM}$ [R$_E$]')
    ax.set_ylabel('Z$_{GSM}$ [R$_E$]')
    ax.text(0.5, 1.01, ti_str, transform=ax.transAxes)

if __name__ == '__main__':
    fig, ax = plt.subplots()
    camera = celluloid.Camera(fig)
    files = glob.glob('RESULTS/y*')
    files.sort()

    for fn in files[:42]:
        makeplot(fn, ax)
        camera.snap()
    animation = camera.animate()
    animation.save('test_anim_x-z-plane.gif')