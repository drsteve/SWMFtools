#!/usr/bin/env python

import os
import errno
import sys
import shutil
import subprocess
import time
import glob
import argparse
from contextlib import contextmanager

defaults = {'nNodes': 12, #12 node (432 core) default
            'jobLength': '7:00:00', #default job length in hh:mm:ss
            'execDir': '/usr/projects/carrington/src/SWMF_grizzly_BATS_Rid_RCM',
            'noSubmit': False,
            'ramscb': False
            }

@contextmanager
def cd(newdir):
    '''Context-managed chdir; changes back to original directory on exit or failure'''
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

def writeArbitrary():
    '''Manager for distributing work when mixing MPI and OpenMP for RAMSCB'''
    with open('arbitrary.pl', 'w') as fh:
        fh.write("#!/usr/bin/perl\n")
        fh.write("my @tasks = split(',', $ARGV[0]);\n")
        #fh.write("my @nodes = `scontrol show hostnames $SLURM_JOB_NODELIST`;\n")
        fh.write("my @nodes = `scontrol show hostnames`;\n")
        fh.write("my $node_cnt = $#nodes + 1;\n")
        fh.write("my $task_cnt = $#tasks + 1;\n")
        fh.write("if ($node_cnt < $task_cnt) {\n")
        fh.write("        print STDERR 'ERROR: You only have $node_cnt nodes, but requested layout on $task_cnt nodes';\n")
        fh.write("        $task_cnt = $node_cnt;\n")
        fh.write("}\n")
        fh.write("my $cnt = 0;\n")
        fh.write("my $layout;\n")
        fh.write("foreach my $task (@tasks) {\n")
        fh.write("        my $node = $nodes[$cnt];\n")
        fh.write("        last if !$node;\n")
        fh.write("        chomp($node);\n")
        fh.write("        for(my $i=0; $i < $task; $i++) {\n")
        fh.write("                $layout .= ',' if $layout;\n")
        fh.write("                $layout .= $node;\n")
        fh.write("        }\n")
        fh.write("        $cnt++;\n")
        fh.write("}\n")
        fh.write("print $layout;")

def writeJobScript(runnum, options):
    nNodes = options.nNodes
    timearg = options.jobLength
    with open('job_script.sh', 'w') as fh:
        fh.write("#!/bin/tcsh\n")
        fh.write("#SBATCH --time={0}\n".format(timearg))
        fh.write("#SBATCH --nodes={0}\n".format(nNodes))
        fh.write("#SBATCH --no-requeue\n")
        fh.write("#SBATCH --job-name=SWMF_{0}\n".format(runnum))
        fh.write("#SBATCH -o slurm%j.out\n")
        fh.write("#SBATCH -e slurm%j.err\n")
        fh.write("#SBATCH --qos=standard\n")
        # Update allocation name to appropriate SLURM group
        fh.write("#SBATCH --account=allocation_name\n")
        # Update email to appropriate domain
        fh.write("#SBATCH --mail-user={0}@email.provider\n".format(os.environ['USER']))
        fh.write("#SBATCH --mail-type=FAIL\n")
        fh.write("\n")
        # Update modules for however SWMF was compiled
        fh.write("module load gcc/6.4.0 openmpi/2.1.2 idl/8.5.1\n")
        if options.ramscb:
            fh.write("module load hdf5-parallel/1.8.16 netcdf-h5parallel/4.4.0\n")
            fh.write("setenv LD_LIBRARY_PATH /usr/projects/carrington/lib:${LD_LIBRARY_PATH}\n")
            fh.write("\n") 
            fh.write("srun -m arbitrary -n {0} -w `perl arbitrary.pl {1}` ./SWMF.exe\n".format(nNodes*36-35,','.join(['1']+['36']*(nNodes-1))))
            writeArbitrary()
        else:
            fh.write("\n") 
            fh.write("srun -n {0} ./SWMF.exe\n".format(nNodes*36))


if __name__=='__main__':
    # Define a command-line option parser and add the options we need
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--version', action='version', version='%prog Version 0.99 (Jan 14, 2019)')

    parser.add_argument('-n', '--nodes', dest='nNodes', type=int,
                        help='Sets number of requested nodes for Slurm job. Default is 12.')

    parser.add_argument('-l', '--length', dest='jobLength',
                        help='Sets requested job length (hh:mm:ss). Default is "7:00:00"')

    parser.add_argument('-e', '--exec_dir', dest='execDir',
                        help='Directory of SWMF installation to use. Default is "/usr/projects/carrington/src/SWMF_grizzly_BATS_Rid_RCM"')

    parser.add_argument('--ramscb', dest='ramscb', action='store_true', help='Enables RAM-SCB-specific settings')

    parser.add_argument('--no_submit', dest='noSubmit', action='store_true',
                        help='Disables the job submission feature of the script. Default is to submit the job')

    parser.add_argument('runname', help='Run name, used to set location for retrieving run files and setting job name.')

    # Parse the args that were (potentially) given to us on the command line
    options = parser.parse_args()
    # set defaults where values aren't provided
    for key in defaults:
        if (not hasattr(options, key)) or (options.__dict__[key] is None):
            options.__dict__[key] = defaults[key]
    user = os.environ['USER']

    run_name = 'run_grizzly_{0}'.format(options.runname)
    with cd(options.execDir):
        subprocess.call(['make', 'rundir'])

    # Update to refelct location of model input directories
    inloc = '/usr/projects/carrington/modelInputs/{0}'.format(options.runname)
    # Update path to HPC cluster scratch directory
    runloc = '/path/to/scratch/{0}'.format(user)
    shutil.move(os.path.join(options.execDir, 'run'), os.path.join(runloc, run_name))
    inputfiles = glob.glob(os.path.join(inloc,'*'))
    for fname in inputfiles:
        try:
            namepart = os.path.split(fname)[-1]
            shutil.copytree(fname, os.path.join(runloc, run_name, namepart))
        except OSError as exc:
            if exc.errno == errno.ENOTDIR:
                shutil.copy(fname, os.path.join(runloc, run_name))
            else:
                print(fname)
                raise OSError

    with cd(os.path.join(runloc, run_name)):
       if os.path.isfile('PARAM.in.start'):
           try:
               os.unlink('PARAM.in')
           except:
               pass
           os.symlink('PARAM.in.start', 'PARAM.in')
       writeJobScript(run_name, options)
       if not options.noSubmit:
           time.sleep(6)
           subprocess.call(['sbatch', 'job_script.sh'])
