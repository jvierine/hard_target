#!/usr/bin/env python

import numpy as n
import matplotlib.pyplot as plt
import stuffr

import scipy.io as sio
import glob

def determine_t0(a,dt=640.0*20e-3):
    t0=int(a["d_parbl"][0][42]*1000000)
    dt=640*20000
    tnow=t0+int(n.round((1e6*a["d_parbl"][0][10]-t0)/dt)-1)*dt
    return(tnow)

def determine_t0_24(a,dt=640.0*20e-3):
    t0=int(a["d_parbl"][0][42]*1000000)
    dt=640*20000
    tnow=t0+int(n.round((1e6*a["d_parbl"][0][10]-t0)/dt)-1)*dt
    return(tnow,tnow+dt)

def determine_t0_25(a,dt=640.0*20e-3):
    t0=int(a["d_parbl"][0][42]*1000000)
    dt=2*640*10000
    tnow=t0+int(n.round((1e6*a["d_parbl"][0][10]-t0)/dt)-1)*dt
    return(tnow,tnow+dt)

# fl = glob.glob("*.mat")
# fl.sort()
# tmax=640.0*20e-3
# prevt=0
# for f in fl:
#     a=sio.loadmat(f)
#     freq=a["d_parbl"][0][54]
#     tnow=determine_t0(a)
#     print(tnow)
#     print(stuffr.unix2datestr(tnow/1e6))
#     print(tnow-prevt)
#     prevt=tnow
# #    print(stuffr.unix2datestr(a["d_parbl"][0][10]))
#     A=n.zeros([640,200])
#     idx=n.arange(20000,dtype=n.int64)
#     for i in range(640):
#         A[i,:]=n.abs(stuffr.decimate(a["d_raw"][idx+20000*i][:,0],dec=100))**2.0
#     rvec=n.arange(200)*100*0.15
#     tvec=n.arange(640L)*20000L
#     plt.pcolormesh(rvec,tvec,10.0*n.log10(A),vmin=0,vmax=80.0)
#     plt.title("%s to %s"%(stuffr.unix2datestr(a["d_parbl"][0][10]-tmax),stuffr.unix2datestr(a["d_parbl"][0][10])))
#     plt.colorbar()
#     plt.show()
