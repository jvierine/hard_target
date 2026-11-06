import ctypes as C
import numpy as n
import time
import matplotlib.pyplot as plt
from. import __path__
# compile with Makefile
_fmed = n.ctypeslib.load_library(__path__[0]+'/libgmfgpu', './Ccode')
_fmed.gmf.restype = C.c_int
_fmed.gmf.argtypes = [C.POINTER(C.c_float), C.c_int, C.POINTER(C.c_float), C.c_int, C.POINTER(C.c_float), C.c_int, C.POINTER(C.c_float), C.c_int, C.c_int, C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_int]

def gmf(z_tx, z_rx, acc_phasors, rgs, dec, gmf_vec, gmf_dc_vec, v_vec, a_vec, rank=0):
    print("Using GPU")
    txlen=int(len(z_tx))
    rxlen=len(z_rx)
    a=_fmed.gmf(z_tx.ctypes.data_as(C.POINTER(C.c_float)),
                txlen,
                z_rx.ctypes.data_as(C.POINTER(C.c_float)),
                rxlen,
                acc_phasors.ctypes.data_as(C.POINTER(C.c_float)),
                int(acc_phasors.shape[0]),
                rgs.ctypes.data_as(C.POINTER(C.c_float)),
                len(rgs),
                int(dec),
                gmf_vec.ctypes.data_as(C.POINTER(C.c_float)),
                gmf_dc_vec.ctypes.data_as(C.POINTER(C.c_float)),
                v_vec.ctypes.data_as(C.POINTER(C.c_float)),
                a_vec.ctypes.data_as(C.POINTER(C.c_float)),
                rank)
    return(1)

def basic_test():
    z_tx=n.zeros(10000,dtype=n.complex64)
    z_rx=n.zeros(12000,dtype=n.complex64)    
    for i in range(10):
        z_tx[(i*1000):(i*1000+20)]=1.0
        z_rx[(i*1000+500):(i*1000+(500+20))]=0.5 # simulated "echo"
    
    dec=10
    acc_phasors=n.zeros([20,1000],dtype=n.complex64)
    acc_phasors[0,:]=1.0
    rgs=n.zeros(1000,dtype=n.float32)#arange(700,dtype=n.int64)
    for ri in range(len(rgs)):
        rgs[ri]=ri
    n_r=len(rgs)
    gmf_vec=n.zeros(n_r,dtype=n.float32);
    gmf_dc_vec=n.zeros(n_r,dtype=n.float32);
    v_vec=n.zeros(n_r,dtype=n.float32);
    a_vec=n.zeros(n_r,dtype=n.float32);
    cput0=time.time()
    for i in range(20):
        gmf(z_tx,z_rx,acc_phasors,rgs,dec,gmf_vec,gmf_dc_vec,v_vec,a_vec)
    cput1=time.time()
    plt.plot(gmf_vec)
    plt.show()
    plt.plot(gmf_dc_vec)
    plt.show()
    plt.plot(v_vec)
    plt.show()
    plt.plot(a_vec)
    plt.show()
    print("Execution time %1.2f"%(cput1-cput0))
    ri=n.argmax(gmf_vec)
    print("Rmax %d gmf %1.2g v %1.2f a %1.2f"%(ri,gmf_vec[ri],v_vec[ri],a_vec[ri]))


if __name__ == "__main__":
    basic_test()
