import argparse
import glob
import os
import warnings
from xml.etree import ElementTree
from xml.dom import minidom


def prettify(elem):
    """Pretty-print XML string with indents and newlines
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def genPVD(fnames, outfn):
    """Make PVD file with collection of supplied filenames
    """
    vtkfile = ElementTree.Element('VTKFile', type='Collection', version='0.1')
    collection = ElementTree.SubElement(vtkfile, 'Collection')
    for idx, fname in enumerate(fnames):
        ElementTree.SubElement(collection, 'DataSet', timestep=f'{idx}', file=fname)
    outstr = prettify(vtkfile)
    with open(outfn, 'w') as fh:
        fh.write(outstr)


def filterFilename(fname, keeppaths=True):
    if keeppaths:
        return fname
    else:
        return os.path.split(fname)[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Paraview data file (.pvd) to group vtp files for multiple times')
    parser.add_argument('-p', '--preservepaths', action='store_true')
    parser.add_argument('-o', '--output', default="Bats2d.pvd", help='Output filename, default is Bats2d.pvd')
    parser.add_argument('inputfiles', nargs='+', help='Input files (can use wildcards, shell will expand)\n'+
                                                      'e.g., BBF_3D/GM/y*065[1-9][024]0*vtp')
    args = parser.parse_args()

    usefiles = []
    warnings.simplefilter("always")
    
    for fncand in args.inputfiles:
        if not os.path.isfile(fncand):
            tmp = sorted(glob.glob(fncand))
            if not tmp:
                warnings.warn("Search pattern '{}' did not match any files. Skipping.".format(fncand))
            usefiles.extend([filterFilename(fn, args.preservepaths) for fn in tmp])
        else:
            usefiles.append(filterFilename(fncand, args.preservepaths))
    genPVD(usefiles, args.output)
