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


def mergeComponents(vecx, vecy, vecz, zeroy=False, zeroz=False):
    '''Return 3-vectors merged from compnents

    Parameters
    ----------
    vecx : array
        X-components of vector
    vecy : array
        Y-components of vector
    vecz : array
        Z-components of vector
    
    Other Parameters
    ----------------
    zeroy : bool
        Default False. If True, set Y-components to zero.
    zeroz : bool
        Default False. If True, set Z-components to zero.
    '''
    vecy = vecy if zeroy else np.zeros(len(vecy))
    vecz = vecz if zeroz else np.zeros(len(vecz))
    combined = np.c_[vecx, vecy, vecz]
    return combined


def convertOneFile(fname, triangulate=True, make_vectors=True, keep_components=False):
    '''Convert a single file from BATSRUS 2D to VTP

    Parameters
    ----------
    fname : str
        Filename to convert to VTP
    
    Other Parameters
    ----------------
    triangulate : bool
        If True, return a 2D Delaunay triangulation of the data. If False
        just return the point mesh. Default is True.
    make_vectors : bool
        If True, combine vector state variables in VTK vectors. Default is True.
    keep_components : bool
        If True, keep components of vector state variables as scalars in output files.
        Default is False. Warning: Setting this as False while make_vectors is False
        will remove vector quantities from the output VTP files.
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
    vec_dict = {'zeroy': False, 'zeroz': False}
    if 'y' in data and 'z' not in data:
        ys = data['y']
        zs = np.zeros(len(xs))
        vec_dict['zeroz'] = True
    elif 'y' not in data and 'z' in data:
        ys = np.zeros(len(xs))
        zs = data['z']
        vec_dict['zeroy'] = True
    else:  # Input data is probably 3D, but this should still work (untested)
        ys = data['y']
        zs = data['z']

    points = np.c_[xs, ys, zs]
    point_cloud = pv.PolyData(points)

    if 'jx' in data:
        if keep_components:
            point_cloud["jx"] = data["jx"]
            point_cloud["jy"] = data["jy"]
            point_cloud["jz"] = data["jz"]
        if make_vectors:
            point_cloud['jvec_inplane'] = mergeComponents(data["jx"], data["jy"], data["jz"], **vec_dict)
    if 'bx' in data:
        if keep_components:
            point_cloud["bx"] = data["bx"]
            point_cloud["by"] = data["by"]
            point_cloud["bz"] = data["bz"]
        if make_vectors:
            point_cloud['bvec_inplane'] = mergeComponents(data["bx"], data["by"], data["bz"], **vec_dict)
    if 'ux' in data:
        if keep_components:
            point_cloud["ux"] = data["ux"]
            point_cloud["uy"] = data["uy"]
            point_cloud["uz"] = data["uz"]
        if make_vectors:
            point_cloud['uvec_inplane'] = mergeComponents(data["ux"], data["uy"], data["uz"], **vec_dict)
    if 'rho' in data:
        point_cloud["rho"] = data["rho"]
    if 'p' in data:
        point_cloud["p"] = data["p"]

    # Convert from points mesh to connected mesh using triangulation
    if triangulate:
        point_cloud.delaunay_2d(tol=1e-05, alpha=0.0, offset=1.0, bound=False, inplace=True, edge_source=None, progress_bar=False)

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
