from collections import OrderedDict
import vtk
import spacepy.datamodel as dm

#load BATL (HDF5) file
data = dm.fromHDF5('3d__mhd_2_n00009346.batl')
#set output filename (will also create a folder sans extension)
fname = 'test_vtk.vth' #vth is expected extension for hierarchical data sets like AMR
#TODO: use option parser to get filenames (incl. wildcards) for in/out
#TODO: also do checking of extension, is file, does directory exist, etc.

#find number of refine levels and Nblocks at each level
level = OrderedDict()
nrlev = []
for idx, lev in enumerate(list(set(data['refine level']))):
    level[lev] = idx
    nrlev.append((data['refine level']==lev).sum())
nlevels = len(level)
levcounter = [0]*nlevels #counter to label each block (per refine level)
#shortcuts, etc.
blockcoords = data['coordinates']
nblocks = blockcoords.shape[0]
bbox = data['bounding box']
bx = data['Bx']
bsize = bx[0].shape[-1]
#Initialize the NonOverlappingAMR object
multiblock = vtk.vtkNonOverlappingAMR()
multiblock.Initialize(nlevels, nrlev)

#iterate over all blocks
for idx in range(nblocks):
    dx = ((bbox[idx])[0,1]-(bbox[idx])[0,0])/bsize #get resolution
    grid = vtk.vtkUniformGrid() #each block is a uniform grid
    blockcentre = blockcoords[idx]
    blockcentre -= dx*4 #offset from centre to block corner ("origin")
    grid.SetOrigin(*blockcentre)
    grid.SetSpacing(dx, dx, dx)
    grid.SetDimensions(bsize+1, bsize+1, bsize+1) # number of points in each direction
    ncells = grid.GetNumberOfCells()
    #add data [Bx, By, Bz, jx, jy, jz, Rho, P] if present
    if 'jx' in data and 'jy' in data and 'jz' in data:
        jtotarray = vtk.vtkDoubleArray()
        jtotarray.SetNumberOfComponents(3) # 1 for a scalar; 3 for a vector
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
        Utotarray.SetNumberOfComponents(3) # 1 for a scalar; 3 for a vector
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
        Btotarray.SetNumberOfComponents(3) # 1 for a scalar; 3 for a vector
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
        rhoarray.SetNumberOfComponents(1) # 1 for a scalar; 3 for a vector
        rhoarray.SetNumberOfTuples(grid.GetNumberOfCells())
        for ind, val in enumerate(data['Rho'][idx].ravel()):
            rhoarray.SetValue(ind, val)
        grid.GetCellData().AddArray(rhoarray)
        rhoarray.SetName('Rho')
    if 'P' in data:
        parray = vtk.vtkDoubleArray()
        parray.SetNumberOfComponents(1) # 1 for a scalar; 3 for a vector
        parray.SetNumberOfTuples(grid.GetNumberOfCells())
        for ind, val in enumerate(data['P'][idx].ravel()):
            parray.SetValue(ind, val)
        grid.GetCellData().AddArray(parray)
        parray.SetName('P')

    #add block to multiblock set
    lev = level[data['refine level'][idx]]
    levidx = levcounter[lev]
    multiblock.SetDataSet(lev, levidx, grid)
    levcounter[lev] += 1 #increment block counter for given refine level

#set up writer for binary XML output
writer = vtk.vtkXMLUniformGridAMRWriter()
#writer.SetDataModeToAscii()
writer.SetFileName(fname)
writer.SetInputData(multiblock)
writer.Write()
