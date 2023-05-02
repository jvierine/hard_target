#!/usr/bin/env python
#
# This converts EISCAT space debris experiment data format to Digital RF
# Also writes necessary metadata into a .h5 file.
#
#
import glob
import scipy.io as sio
import matplotlib.pyplot as plt
import digital_rf as drf
import os
import numpy as n
import h5py
import stuffr
import sys
import bz2

import gmf_opts as go

# import from this directory. this is needed if this is a package
import parbl

if __name__ == "__main__":


    if len(sys.argv) == 2:
        print(sys.argv[1])
        conf=go.gmf_opts(sys.argv[1])
    else:
        print("Provide configuration file as command line option")
        exit(0)
        
    print(conf.eiscat_dir)

    
    write_drf=True
    
#    if len(sys.argv) != 2:
 #       print("specify path of eiscat data. e.g.,")
  #      print("convert_eiscat2drf.py /scratch/data/juha/eiscat/2021.11.23/leo_bpark_2.1u_NO@uhf")
   #     exit(1)
        
    idir=conf.eiscat_dir#sys.argv[1]
    odir=conf.drf_dir
    os.system("mkdir -p %s"%(odir))
    
    # uhf is the name of the raw voltaga channel
    odir1="%s/uhf"%(odir)
    os.system("mkdir -p %s"%(odir1))

    # write metadata here
    metadata_file = "%s/eiscat_metadata.h5"%(odir)
    
    # how many samples in file. 640 ipps. eiscat leo experiment specific number!
    L=640*20000
    
    # look for all data files
    # warning not Y3K compatible code
    fl=glob.glob("%s/2*/*.mat.bz2"%(idir))
    fl.sort()
    
    a=sio.loadmat(bz2.open(fl[0]))
    t0,t1=parbl.determine_t0_24(a)

    print(t0)
    if write_drf:
        # create digital rf writer
        w=drf.DigitalRFWriter(odir1, n.complex64, 3600, 1000, t0, 1000000, 1, "uhf", compression_level=0, checksum=False, is_complex=True, num_subchannels=1, is_continuous=True, marching_periods=True)
    
    # ipp
    ipp=20000
    ipp_idx=n.arange(ipp)
    t_prev=t0
    tvec=n.zeros(len(fl))
    pvec=n.zeros(len(fl))

    # also create .h5 files containing metadata about power and pointing direction
    eiscat_time = []
    eiscat_elevation = []
    eiscat_azimuth = []
    eiscat_txpwr = []
    eiscat_inj_pwr = [] 
    
    # go through all matlab files
    # and feed into drf writer
    for fi,f in enumerate(fl):
        fh = bz2.open(f)
        a=sio.loadmat(fh)
        
        print(a["d_parbl"].shape)
        print("time %1.2f"%(a["d_parbl"][0,10]))
        print("az %1.2f"%(a["d_parbl"][0,8]))
        print("el %1.2f"%(a["d_parbl"][0,9]))
        print("pwr %1.2f"%(a["d_parbl"][0,7]))
        print("inj pwr %1.2f"%(a["d_parbl"][0,20]))        

        eiscat_time.append(a["d_parbl"][0,10])
        eiscat_azimuth.append(a["d_parbl"][0,8])
        eiscat_elevation.append(a["d_parbl"][0,9])
        eiscat_txpwr.append(a["d_parbl"][0,7])
        eiscat_inj_pwr.append(a["d_parbl"][0,20])

        # write metadata (all of it into one file)
        # keep updating file with latest data
        ho=h5py.File(metadata_file,"w")
        ho["time"]=eiscat_time
        ho["azimuth"]=eiscat_azimuth
        ho["elevation"]=eiscat_elevation
        ho["tx_pwr"]=eiscat_txpwr
        ho["inj_pwr"]=eiscat_inj_pwr
        ho.close()
              
        if write_drf:
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
            if False:
                plt.plot(z.real)
                plt.plot(z.imag)
                plt.show()
    
            L=len(z)
            z_txp=n.zeros(20000)
            if L == 12800000:
                w.rf_write(z)
                
        t_prev=t0
        fh.close()

