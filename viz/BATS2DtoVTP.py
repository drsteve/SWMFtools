import argparse
from collections import OrderedDict
import datetime as dt
import os
import re
import sys
import numpy as np
import spacepy.datamodel as dm
import spacepy.pybats.bats as bts
import pyvista as pv
import vtk


def timeFromFilename(outname):
    '''Get timestamp from SWMF/BATSRUS output file

    Supports "_e" and "_t" naming conventions.
    If both are present, it will use "_e" by preference.
    Precision is currently to the nearest second.
    '''
    tsrch = ".*_t(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})"
    esrch = ".*_e(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})"
    if '_t' in outname or '_e' in outname:
        srch = esrch if '_e' in outname else tsrch
        tinfo = re.search(srch, outname).groups()
        tinfo = [val for val in map(int, tinfo)]
    else:
        raise NotImplementedError('Only "_t" and "_e" format times are supported')

    dtobj = dt.datetime(*tinfo)
    return dtobj


def convertOneFile(fname):
    '''Convert a single file from BATSRUS 2D to VTP
    '''
    # load BATSRUS 2D file
    data = bts.Bats2d(fname)
    outname = os.path.splitext(fname)[0]
    # set output filename
    fname = '.'.join((outname, 'vtp'))

    # Initialize the VTK object
    # Should work on y=0 and z=0 slices, probably also 3D IDL.
    # Won't work on x=0 or arbitrary slice... (yet)
    xs = data['x']
    if 'y' in data and 'z' not in data:
        ys = data['y']
        zs = np.zeros(len(xs))
    elif 'y' not in data and 'z' in data:
        ys = np.zeros(len(xs))
        zs = data['z']
    else:  # Input data is probably 3D, but this should still work (untested)
        ys = data['y']
        zs = data['z']

    points = np.c_[xs, ys, zs]
    point_cloud = pv.PolyData(points)

    if 'jx' in data:
        point_cloud["jx"] = data["jx"]
        point_cloud["jy"] = data["jy"]
        point_cloud["jz"] = data["jz"]
    if 'bx' in data:
        point_cloud["bx"] = data["bx"]
        point_cloud["by"] = data["by"]
        point_cloud["bz"] = data["bz"]
    if 'ux' in data:
        point_cloud["ux"] = data["ux"]
        point_cloud["uy"] = data["uy"]
        point_cloud["uz"] = data["uz"]
    if 'rho' in data:
        point_cloud["rho"] = data["rho"]
    if 'p' in data:
        point_cloud["p"] = data["p"]

    # write to binary XML output
    point_cloud.save(fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert BATS-R-US 2D IDL files to VTP (VTK point data)')
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('inputfiles', nargs='+', help='Input files (can use wildcards, shell will expand)')
    args = parser.parse_args()
    silent = args.silent
    for batlname in args.inputfiles:
        bname, ext = os.path.splitext(batlname)
        if os.path.isdir(batlname):
            if not silent: print('Input file {0} appears to be a directory. Skipping'.format(batlname))
        elif not ext.lower()=='.out':
            if not silent: print('Input file {0} may not be a BATS-R-US file. Skipping'.format(batlname))
        else:
            try:
                convertOneFile(batlname)
            except IOError:
                if not silent: print("Unexpected error processing {0}:".format(batlname), sys.exc_info()[0])
