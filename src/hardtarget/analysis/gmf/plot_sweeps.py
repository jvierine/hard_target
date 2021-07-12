import numpy as n
import h5py
import matplotlib.pyplot as plt


def plot_snr_sweep():
    h=h5py.File("snr_sweep.h5","r")
    print(h.keys())
    plt.subplot(121)
    plt.semilogy(h["snrs"][()],n.abs(h["range_err"][()]),".")
    plt.ylabel("Range error (m)")
    plt.xlabel("SNR (dB)")    
    
    plt.subplot(122)
    plt.semilogy(h["snrs"][()],n.abs(h["vel_err"][()]),".")
    plt.ylabel("Range-rate error (m/s)")
    plt.xlabel("SNR (dB)")    
    plt.show()
    h.close()
    
def plot_ipp_sweep():
    h=h5py.File("n_ipp_sweep.h5","r")
    print(h.keys())
    plt.loglog(h["n_ipps"][()],h["gmf_time"][()])
    #plt.semilogy(h["n_ipps"][()],h["finetune_time"][()])
    plt.xlabel("Coherent integration length (number of IPPs)")
    plt.ylabel("CPU time / Wallclock time")
    plt.grid()
    plt.show()
    h.close()


plot_snr_sweep()
plot_ipp_sweep()
