
from . import gmf_opts as go
import glob
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as n
import scipy.constants as c

def get_noise_floor(conf):
    fl=glob.glob("%s/*/gmf*.h5"%(conf.output_dir))
    fl.sort()
    floors=[]
    for f in fl:
        print(f)
        h=h5py.File(f,"r")
        gmf=h["gmf"][()]
        floors.append(n.nanmedian(gmf,axis=0))
        h.close()
    floors=n.array(floors)
    return(n.nanmedian(floors,axis=0))


def plot_gmfs(conf,nf):
    fl=glob.glob("%s/*/gmf*.h5"%(conf.output_dir))
    fl.sort()
    print(len(fl))
    for f in fl:
        print(f)
        h=h5py.File(f,"r")
        gmf=n.copy(h["gmf"][()])
        v=n.copy(h["v"][()])
        a=n.copy(h["a"][()])                
        
        for i in range(gmf.shape[0]):
            gmf[i,:]=(gmf[i,:]-nf)/nf
#            plt.plot(gmf[i,:])
 #           plt.show()

        if n.max(gmf) > 10.0:
            plt.subplot(221)
            plt.plot(n.max(gmf,axis=1))
            plt.subplot(222)
            dr=c.c/conf.sample_rate/2.0/1e3
            plt.plot(dr*conf.rgs[n.argmax(gmf,axis=1)],".")
            plt.subplot(223)
            
            vels=[]
            for i in range(gmf.shape[0]):
                mi=n.argmax(gmf[i,:])
                vels.append(v[i,mi])
            
            plt.plot(vels,".")
            plt.subplot(224)            
            ass=[]
            for i in range(gmf.shape[0]):
                mi=n.argmax(gmf[i,:])
                ass.append(a[i,mi])
            
            plt.plot(ass,".")
            plt.show()
        h.close()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        print(sys.argv[1])
        conf=go.gmf_opts(sys.argv[1])
    else:
        print("Provide configuration file as command line option")
        exit(0)
    nf=get_noise_floor(conf)
    nf[nf<1]=1.0
    plt.plot(nf)
    plt.show()
    plot_gmfs(conf,nf)

