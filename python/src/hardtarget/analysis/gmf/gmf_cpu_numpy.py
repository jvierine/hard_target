#!/usr/bin/env pyth
#
# Example of range-Doppler-acceleration matched filter (Generalized Match Function)
#
# # example1
#i0=1463100914401305
#rg0=2000#4400
## example 2
##i0=1463106122981305
#rg0=5425
# example 3 (range aliased echo!)
#i0=1463107264431305
#rg0=8145+10000


import os
import time

import h5py

import numpy as n
import matplotlib.pyplot as plt
import scipy.fftpack as fft
import scipy.constants as c
import scipy.optimize as sio

import digital_rf as drf
import stuffr
#import pyfftw

from ..utilities import read_vector_c81d


def analyze_ipps(d,i0,o,mode=0,plott=False):
    print("Using numpy")
    # read data vector with n_ipps, and a little extra
    z_tx=read_vector_c81d(d,i0,(o.n_ipp+o.n_extra)*o.ipp,o.tx_channel)
    z_rx=read_vector_c81d(d,i0,(o.n_ipp+o.n_extra)*o.ipp,o.rx_channel)

    if plott:
        # plot echo
        plt.plot(z_tx.real)
        plt.plot(z_tx.imag)
        plt.title("Echo")
        plt.show()

    # make a separate copy to hold transmit pulse, and the echo
    z_tx=n.copy(z_tx)
    z_rx=n.copy(z_rx)

    # clean ground clutter, get separate transmit waveform and echo vectors
    z_tx=z_tx*o.tx_stencil
    z_rx=z_rx*o.rx_stencil

    # truncate the tx vector to be exactly the length of n_ipp
    z_tx=z_tx[0:(o.n_ipp*o.ipp)]

    # conjugate, so that when matched filtering, it will cancel out phase of transmit waveform.
    z_tx=n.conj(z_tx)

    if plott:
        # plot tx
        plt.plot(z_tx.real)
        plt.plot(z_tx.imag)
        plt.title("Only tx")
        plt.show()

        # plot echo
        plt.plot(z_rx.real)
        plt.plot(z_rx.imag)
        plt.title("Only echo")
        plt.show()

    cput0=time.time()
    max_r=0
    max_v=0
    max_a=0
    max_mf=0
    gmf_vec=n.zeros(o.n_range_gates,dtype=n.float32)
    v_vec=n.zeros(o.n_range_gates,dtype=n.float32)
    a_vec=n.zeros(o.n_range_gates,dtype=n.float32)
    gmf_dc_vec=n.zeros(o.n_range_gates,dtype=n.float32)

    # implement this in C
    #
    # gmf.mf(z_tx, o.n_fft, z_rx,len(z_rx), o.acc_phasors, o.n_accs, o.rgs,o.fdec,gmf_vec,gmf_dc_vec,v_vec,a_vec)
    GA=n.zeros([len(o.accs),len(o.rgs)])
    GV=n.zeros([int(o.n_fft/o.frequency_decimation),len(o.rgs)])
    for ri,rg in enumerate(o.rgs):
        # range matching echo*conj(tx)
        echo=stuffr.decimate(z_rx[rg:(rg+o.n_fft)]*z_tx,dec=o.frequency_decimation)

        # go through all accelerations
        for ai,a in enumerate(o.accs):
            # go through all doppler shifts with FFT (this is a grid search of
            # all possible doppler velocities)
            gmf=n.abs(fft.fft(o.acc_phasors[ai,:]*echo,len(echo)))**2.0
            mi=n.argmax(gmf)
            GA[ai,ri]=gmf[mi]

            if ai==0:
                gmf_dc_vec[ri]=gmf[0]

            if gmf[mi]>gmf_vec[ri]:
                gmf_vec[ri]=gmf[mi]
                v_vec[ri]=o.range_rates[mi]
                a_vec[ri]=a
#    plt.pcolormesh(o.rgs,o.accs,GA)
#    plt.show()


    if plott:
        plt.plot(gmf_vec)
        plt.show()
    cput1=time.time()
    print("time %1.2f cpu/real %1.2f"%(cput1-cput0,(cput1-cput0)/(o.n_ipp*o.ipp/o.sample_rate)))
    mri=n.argmax(gmf_vec)
    print("GMF=%1.2g r_max=%1.2f (km) vel_max=%1.2f (km/s) a_max=%1.2f (m/s**2)"%(o.frequency_decimation*n.max(gmf_vec),o.ranges[mri],v_vec[mri]/1e3,a_vec[mri]))
    return(gmf_vec,gmf_dc_vec,v_vec,a_vec,1)

def analyze_ipps_fine(d,
                      o,
                      i0,
                      r0=1000e3,
                      rlim=[-100,100],
                      n_r=3,
                      vlim=[-10.0,10.0],
                      n_v=11,
                      v0=2e3,
                      a0=0.0,
                      alim=[0,100.0],
                      n_a=40,
                      noise_pwr=1.0,
                      plott=False):

    # read data vector with n_ipps, and a little extra
    z_tx=read_vector_c81d(d,i0,(o.n_ipp+o.n_extra)*o.ipp,o.tx_channel)
    z_rx=read_vector_c81d(d,i0,(o.n_ipp+o.n_extra)*o.ipp,o.rx_channel)

    # make a separate copy to hold transmit pulse, and the echo
    z_tx=n.copy(z_tx)
    z_rx=n.copy(z_rx)

    # clean ground clutter, get separate transmit waveform and echo vectors
    z_tx=z_tx*o.tx_stencil
    z_rx=z_rx*o.rx_stencil

    # truncate the tx vector to be exactly the length of n_ipp
    z_tx=z_tx[0:(o.n_ipp*o.ipp)]

    # conjugate, so that when matched filtering, it will cancel out phase of transmit waveform.
    z_tx=n.conj(z_tx)

    if plott:
        # plot tx
        plt.plot(z_tx.real)
        plt.plot(z_tx.imag)
        plt.title("Only tx")
        plt.show()

        # plot echo
        plt.plot(z_rx.real)
        plt.plot(z_rx.imag)
        plt.title("Only echo")
        plt.show()

    tau = n.arange(o.n_ipp*o.ipp)/o.sample_rate

    def gmf(x):
        r0=x[0]
        v0=x[1]
        a0=x[2]
        rng = r0 + v0*tau + 0.5*a0*tau**2.0
        phase=n.exp(-1j*n.mod(2.0*n.pi*rng/o.wavelength,2.0*n.pi))
        rng_samples = n.array(n.round(o.sample_rate*rng/c.c),dtype=n.int)
        idx = n.arange(o.n_ipp*o.ipp)
        mf = z_rx[rng_samples + idx]*phase*z_tx
        s=n.abs(n.sum(mf))
        return(-s)

    drs=n.linspace(r0+rlim[0],r0+rlim[1],num=n_r)
    dvs=n.linspace(v0+vlim[0],v0+vlim[1],num=n_v)
    das=n.linspace(alim[0],alim[1],num=n_a)
    best=0.0
    xhat=[0,0,0]
    for ri in range(n_r):
        r0t=drs[ri]
        for vi in range(n_v):
            v0t=dvs[vi]
            for ai in range(n_a):
                a0t=das[ai]
                test=-gmf([r0t,v0t,a0t])
                if test > best:
                    best=test
                    xhat=[r0t,v0t,a0t]
                    print("snr %1.2f (dB) r %1.5f v %1.5f a %1.5f"%(10.0*n.log10((best-n.sqrt(noise_pwr))**2.0/noise_pwr),r0t,v0t,a0t))

    print("fmin")
    xhat=sio.fmin(gmf,xhat)
    xhat=sio.fmin(gmf,xhat)
    xhat=sio.fmin(gmf,xhat)

    best_gmf=-gmf(xhat)
    best_snr=(best_gmf-n.sqrt(noise_pwr))**2.0/noise_pwr
    return([best_snr,xhat[0],xhat[1],xhat[2]])


