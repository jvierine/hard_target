#!/usr/bin/env python


import glob
import scipy.io as sio
import matplotlib.pyplot as plt
import digital_rf as drf
import os
import numpy as n
import h5py
import stuffr
import sys

# import from this directory. this is needed if this is a package
from . import parbl

def main():
    if len(sys.argv) != 2:
        print("specify path of eiscat data. e.g.,")
        print("convert_eiscat2drf.py /scratch/data/juha/eiscat/2021.11.23/leo_bpark_2.1u_NO@uhf")
        exit(1)
        
    idir=sys.argv[1]
#    idir="/scratch/data/juha/eiscat/2021.11.23/leo_bpark_2.1u_NO@uhf"
    odir="%s/drf"%(idir)
    os.system("mkdir -p %s"%(odir))
    odir1="%s/uhf"%(odir)
    os.system("mkdir -p %s"%(odir1))
    
    # how many samples in file. 640 ipps. eiscat leo experiment specific number!
    L=640*20000
    
    # look for all data files
    # warning not Y3K compatible code
    fl=glob.glob("%s/2*/*.mat"%(idir))
    fl.sort()
    
    a=sio.loadmat(fl[0])
    t0,t1=parbl.determine_t0_24(a)
    
    # create digital rf writer
    w=drf.DigitalRFWriter(odir1, n.complex64, 3600, 1000, t0, 1000000, 1, "uhf", compression_level=0, checksum=False, is_complex=True, num_subchannels=1, is_continuous=True, marching_periods=True)
    
    # ipp
    ipp=20000
    ipp_idx=n.arange(ipp)
    t_prev=t0
    tvec=n.zeros(len(fl))
    pvec=n.zeros(len(fl))
    # go through all matlab files
    # and feed into drf writer
    for fi,f in enumerate(fl):
        a=sio.loadmat(f)
        # this one figures out
        t0,t1=parbl.determine_t0_24(a)
        print("n_samp %d"%(t0-t_prev))
        if t0-t_prev != 12800000 and t0-t_prev != 0:
            # if start time is not 12800000 samples more than previous one, we
            # have a missing data file. we must pad zeros into the data 
            n_samp=(t0-t_prev)-12800000
            print("padding zeros %d"%(n_samp))        
            zz=n.zeros(n_samp,dtype=n.complex64)
            w.rf_write(zz)
        
        z=n.array(a["d_raw"][:,0],dtype=n.complex64)
    
        L=len(z)
        z_txp=n.zeros(20000)
        if L == 12800000:
            w.rf_write(z)
        t_prev=t0

