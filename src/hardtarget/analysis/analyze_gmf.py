#!/usr/bin/env python

import h5py
import numpy as n
import os
import sys

import digital_rf as drf
from . import gmf_opts as go
import stuffr
from . import analyze_ipps as g

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    class COMM_WORLD:
        rank = 1
        size = 1
    comm = COMM_WORLD()


def analyze_gmf(conf,
                n_ints=0):
    
    if conf.reanalyze:
        os.system("rm -f %s/*/*.h5"%(conf.output_dir))

    d=drf.DigitalRFReader(conf.data_dirs)
    
    b_rx=d.get_bounds(conf.rx_channel)
    b_tx=d.get_bounds(conf.tx_channel)
    
    print("RX bounds %d-%d TX bounds %d-%d"%(b_rx[0],b_rx[1],b_tx[0],b_tx[1]))
    print("RX bounds %s-%s TX bounds %s-%s"%(stuffr.unix2datestr(b_rx[0]/conf.sample_rate),
                                             stuffr.unix2datestr(b_rx[1]/conf.sample_rate),
                                             stuffr.unix2datestr(b_tx[0]/conf.sample_rate),
                                             stuffr.unix2datestr(b_tx[1]/conf.sample_rate)))
    
    print("Number of parallel processes %d"%(comm.size))

    b=b_rx
    b=[b_rx[0],b_rx[1]]
    if conf.t0 != None:
        b[0]=int(conf.t0*conf.sample_rate)
        

        
    if n_ints==0:
        n_ints=int(n.floor((b[1]-b[0])/(conf.ipp*conf.n_ipp))/conf.num_cohints_per_file)

    # parallel
    for ni in range(comm.rank,n_ints,comm.size):
        fi=ni*conf.ipp*conf.n_ipp*conf.num_cohints_per_file + b[0]        
        print("rank %d %s"%(comm.rank,stuffr.unix2datestr(fi/conf.sample_rate)))
        hdname="%s/%s"%(conf.output_dir,stuffr.sec2dirname(fi/conf.sample_rate))
        fname="%s/gmf-%08d.h5"%(hdname,fi)

        gmf_max=n.zeros([conf.num_cohints_per_file,conf.n_range_gates],dtype=n.float32)
        gmf_dc=n.zeros([conf.num_cohints_per_file,conf.n_range_gates],dtype=n.float32)
        gmf_v=n.zeros([conf.num_cohints_per_file,conf.n_range_gates],dtype=n.float32)
        gmf_a=n.zeros([conf.num_cohints_per_file,conf.n_range_gates],dtype=n.float32)
        gmf_txp=n.zeros(conf.num_cohints_per_file,dtype=n.float32)
    
        if os.path.exists(fname):
            print("skipping %d, file already exists"%(fi))
        else:
            for i in range(conf.num_cohints_per_file):
                i0=fi + i*conf.ipp*conf.n_ipp
                
                gmf_max[i,:], gmf_dc[i,:], gmf_v[i,:], gmf_a[i,:], gmf_txp[i] = g.analyze_ipps(d,i0,conf)
                rgi=n.argmax(gmf_max[i,:])

            os.system("mkdir -p %s"%(hdname))
            print("writing %s"%(fname))
            ho=h5py.File(fname,"w")
            ho["gmf"]=gmf_max
            ho["gmf_dc"]=gmf_dc
            ho["a"]=gmf_a
            ho["v"]=gmf_v
            ho["tx_pwr"]=gmf_txp
            ho["i0"]=i0
            ho.close()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        print(sys.argv[1])
        conf=go.gmf_opts(sys.argv[1])
    else:
        print("Provide configuration file as command line option")
        exit(0)
    print(conf)
    analyze_gmf(conf)
