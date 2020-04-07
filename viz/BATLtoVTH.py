import os
import sys
import argparse
from collections import OrderedDict
import numpy as np
import spacepy.datamodel as dm
try:
    import re
    import datetime as dt
    import spacepy.time as spt
    import spacepy.coordinates as spc
    import spacepy.irbempy as ib
    import pyvista as pv
    irb = True
except ImportError:
    irb = False
import vtk


def timeFromFilename(outname):
    '''
    '''
    tsrch = ".*_t(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})"
    if '_t' in outname:
        tinfo = re.search(tsrch, outname).groups()
        tinfo = [val for val in map(int, tinfo)]
    else:
        raise NotImplementedError('Only "_t" format times currently supported')

    dtobj = dt.datetime(*tinfo)
    return dtobj


def convertOneFile(batlname, add_dipole=False):
    '''Convert a single file from BATL HDF5 to VTH
    '''
    # load BATL (HDF5) file
    data = dm.fromHDF5(batlname)
    outname = os.path.splitext(batlname)[0]
    # set output filename (will also create a folder sans extension)
    fname = '.'.join((outname, 'vth'))  # vth is expected extension for hierarchical data sets like AMR

    # find number of refine levels and Nblocks at each level
    level = OrderedDict()
    nrlev = []
    for idx, lev in enumerate(list(set(data['refine level']))):
        level[lev] = idx
        nrlev.append((data['refine level'] == lev).sum())
    nlevels = len(level)
    levcounter = [0]*nlevels  # counter to label each block (per refine level)
    # shortcuts, etc.
    blockcoords = data['coordinates']
    nblocks = blockcoords.shape[0]
    bbox = data['bounding box']
    bx = data['Bx']
    bsize = bx[0].shape[-1]  # all blocks have the same dimension (8x8x8 default)

    # Initialize the NonOverlappingAMR object
    multiblock = vtk.vtkNonOverlappingAMR()
    multiblock.Initialize(nlevels, nrlev)

    # iterate over all blocks
    for idx in range(nblocks):
        dx = ((bbox[idx])[0, 1]-(bbox[idx])[0, 0])/bsize  # get resolution
        grid = vtk.vtkUniformGrid()  # each block is a uniform grid
        blockcentre = blockcoords[idx]
        blockcentre -= dx*4  # offset from centre to block corner ("origin")
        grid.SetOrigin(*blockcentre)
        grid.SetSpacing(dx, dx, dx)
        grid.SetDimensions(bsize+1, bsize+1, bsize+1)  # number of points in each direction
        ncells = grid.GetNumberOfCells()

        if add_dipole:
            tval = timeFromFilename(outname)
            pvblock = pv.UniformGrid(grid)
            cellcenters = pvblock.cell_centers().points
            bintx1d = np.empty(len(cellcenters))
            binty1d = np.empty(len(cellcenters))
            bintz1d = np.empty(len(cellcenters))
            coords = spc.Coords(cellcenters, 'GSM', 'car')
            tvals = spt.Ticktock([tval]*len(cellcenters), 'UTC')
            Bgeo = ib.get_Bfield(tvals, coords, extMag='0',
                                 options=[1, 1, 0, 0, 5])['Bvec']
            tmp = spc.Coords(Bgeo, 'GEO', 'car', ticks=tvals)
            Bgsm = tmp.convert('GSM', 'car')
            for ind, xyz in enumerate(Bgsm):
                bintx1d[ind] = xyz.x
                binty1d[ind] = xyz.y
                bintz1d[ind] = xyz.z
            bintarray = vtk.vtkDoubleArray()
            bintarray.SetNumberOfComponents(3)
            bintarray.SetNumberOfTuples(grid.GetNumberOfCells())
            for ind in range(len(bintx1d)):
                bintarray.SetTuple3(ind, bintx1d[ind], binty1d[ind], bintz1d[ind])
            grid.GetCellData().AddArray(bintarray)
            bintarray.SetName('Bvec_int')

        # add data [Bx, By, Bz, jx, jy, jz, Rho, P] if present
        if 'jx' in data and 'jy' in data and 'jz' in data:
            jtotarray = vtk.vtkDoubleArray()
            jtotarray.SetNumberOfComponents(3)  # 1 for a scalar; 3 for a vector
            jtotarray.SetNumberOfTuples(grid.GetNumberOfCells())
            jx1d = data['jx'][idx].ravel()
            jy1d = data['jy'][idx].ravel()
            jz1d = data['jz'][idx].ravel()
            for ind in range(len(jx1d)):
                jtotarray.SetTuple3(ind, jx1d[ind], jy1d[ind], jz1d[ind])
            grid.GetCellData().AddArray(jtotarray)
            jtotarray.SetName('jvec')
        if 'Ux' in data and 'Uy' in data and 'Uz' in data:
            Utotarray = vtk.vtkDoubleArray()
            Utotarray.SetNumberOfComponents(3)  # 1 for a scalar; 3 for a vector
            Utotarray.SetNumberOfTuples(grid.GetNumberOfCells())
            ux1d = data['Ux'][idx].ravel()
            uy1d = data['Uy'][idx].ravel()
            uz1d = data['Uz'][idx].ravel()
            for ind in range(len(ux1d)):
                Utotarray.SetTuple3(ind, ux1d[ind], uy1d[ind], uz1d[ind])
            grid.GetCellData().AddArray(Utotarray)
            Utotarray.SetName('Uvec')
        if 'Bx' in data and 'By' in data and 'Bz' in data:
            Btotarray = vtk.vtkDoubleArray()
            Btotarray.SetNumberOfComponents(3)  # 1 for a scalar; 3 for a vector
            Btotarray.SetNumberOfTuples(grid.GetNumberOfCells())
            bx1d = data['Bx'][idx].ravel()
            by1d = data['By'][idx].ravel()
            bz1d = data['Bz'][idx].ravel()
            for ind in range(len(bx1d)):
                Btotarray.SetTuple3(ind, bx1d[ind], by1d[ind], bz1d[ind])
            grid.GetCellData().AddArray(Btotarray)
            Btotarray.SetName('Bvec')
        if 'Rho' in data:
            rhoarray = vtk.vtkDoubleArray()
            rhoarray.SetNumberOfComponents(1)  # 1 for a scalar; 3 for a vector
            rhoarray.SetNumberOfTuples(grid.GetNumberOfCells())
            for ind, val in enumerate(data['Rho'][idx].ravel()):
                rhoarray.SetValue(ind, val)
            grid.GetCellData().AddArray(rhoarray)
            rhoarray.SetName('Rho')
        if 'P' in data:
            parray = vtk.vtkDoubleArray()
            parray.SetNumberOfComponents(1)  # 1 for a scalar; 3 for a vector
            parray.SetNumberOfTuples(grid.GetNumberOfCells())
            for ind, val in enumerate(data['P'][idx].ravel()):
                parray.SetValue(ind, val)
            grid.GetCellData().AddArray(parray)
            parray.SetName('P')

        # add block to multiblock set
        lev = level[data['refine level'][idx]]
        levidx = levcounter[lev]
        multiblock.SetDataSet(lev, levidx, grid)
        levcounter[lev] += 1  # increment block counter for given refine level

    # set up writer for binary XML output
    writer = vtk.vtkXMLUniformGridAMRWriter()
    # writer.SetDataModeToAscii()
    writer.SetFileName(fname)
    writer.SetInputData(multiblock)
    writer.Write()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert BATL (HDF5) files to VTH (Hierarchical VTK)')
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--add-dipole', dest='add_dipole', action='store_true')
    parser.add_argument('inputfiles', nargs='+', help='Input files (can use wildcards, shell will expand)')
    args = parser.parse_args()
    if args.add_dipole and not irb:
        raise ModuleNotFoundError('Dependencies (pyvista, spacepy.irbempy) could not be loaded')
    silent = args.silent
    for batlname in args.inputfiles:
        bname, ext = os.path.splitext(batlname)
        if os.path.isdir(batlname):
            if not silent: print('Input file {0} appears to be a directory. Skipping'.format(batlname))
        elif not ext.lower()=='.batl':
            if not silent: print('Input file {0} may not be a BATL HDF5 file. Skipping'.format(batlname))
        else:
            try:
                convertOneFile(batlname, add_dipole=args.add_dipole)
            except IOError:
                if not silent: print("Unexpected error processing {0}:".format(batlname), sys.exc_info()[0])
