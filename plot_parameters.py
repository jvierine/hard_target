
import gmf_opts as go
import glob
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as n
import scipy.constants as c

def plot_params(conf):
    fl=glob.glob("%s/*/noise*.h5"%(conf.output_dir))
    fl.sort()
    tt=[]
    tk=[]

    hmd=h5py.File("%s/eiscat_metadata.h5"%(conf.drf_dir),"r")
    
    for i in range(len(fl)):
        h=h5py.File(fl[i],"r")
        tt.append(h["t0"][()])
        tk.append(h["T_sys"][()])
        h.close()
    tt=n.array(tt)
    t_hours=(tt-tt[0])/3600
    plt.plot(tt,n.array(tk),label="T_sys (K)")
    plt.plot(hmd["time"][()],hmd["azimuth"][()],".",label="Az (deg)")
    plt.plot(hmd["time"][()],hmd["elevation"][()],".",label="El (deg)")
    plt.plot(hmd["time"][()],hmd["tx_pwr"][()]/1e5,".",label="TX PWR (10$^5$ W)")
    plt.legend()
    plt.show()
    hmd.close()
    
if __name__ == "__main__":
    if len(sys.argv) == 2:
        print(sys.argv[1])
        conf=go.gmf_opts(sys.argv[1])
    else:
        print("Provide configuration file as command line option")
        exit(0)
    plot_params(conf)

