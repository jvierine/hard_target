#!/usr/bin/env python3

import numpy as n
import matplotlib.pyplot as plt
import pickle
import digital_rf as drf
import os
import scipy.interpolate as sint
import scipy.constants as c

from ..utilities import read_vector_c81d
from ..config import Config

class interp_fun:
    def __init__(self,t0,t1,pfs,mints):
        self.t0=t0
        self.t1=t1
        self.pfs=pfs
        self.mints=mints

    def rfun(self,t):
        print(n.min(t))
        idx=n.where( (n.min(t) >= self.t0) & (n.min(t) < self.t1))[0]
        print(idx)
        idx=idx[0]
        o=self.pfs[idx](t-self.mints[idx])
        return(o)

class raw_sim:
    def __init__(self,freq=230e6,pulse_len=2000,bit_len=1000,sr=1000000,ipp=10000,code=False,code_seed=0):

        # radar configuration
        self.freq=freq
        self.pulse_len=int(pulse_len)
        self.bit_len=int(bit_len)
        self.sr=int(sr)
        self.ipp=int(ipp)
        self.n_bits=int(pulse_len/bit_len)
        self.wavelength=c.c/self.freq
        self.code=code
        n.random.seed(code_seed)

    def interp_funs(self,t,r,dt=1.0):
        self.n_t=int(n.floor((n.max(t)-n.min(t))/dt))

        t0 = n.arange(self.n_t)*dt + n.min(t) - 0.1
        t1 = t0 + dt
        t1[-1]=n.max(t)+0.1
        pfs=[]
        mints=[]

        for i in range(self.n_t):
            tidx=n.where( (t > t0[i]) & (t < t1[i]))[0]
            print(i)
            print(tidx)
            print(t0[i])
            print(t1[i])
            mint=n.min(t[tidx])
            p=n.polyfit(t[tidx]-mint,r[tidx],2)
            pf=n.poly1d(p)
            pfs.append(pf)
            mints.append(mint)
        return(interp_fun(t0,t1,pfs,mints))

    def raw_voltage(self,d,dirname="/media/j/ebd77b41-7efd-4238-b6f8-2b17bc33c84c/debris"):

        print("simulating raw voltage")
        os.system("mkdir -p %s"%(dirname))

        states=d["states"]
        n_objects = len(states)
        n_tx = len(d["observations"][0])
        n_rx = len(d["observations"][0][0])

        t = d["t"]

        min_t = n.min(t)
        max_t = n.max(t)
        delta_t = max_t-min_t

        # how many ipps do we simulate
        n_ipp = int(delta_t*self.sr/self.ipp)

        observations = d["observations"]
        n_rx=0
        passes=[]
        for oi in range(n_objects):

            for txi in range(n_tx):

                txo = observations[oi][txi]
                n_rx =len(txo)

                for rxi in range(n_rx):

                    rxo = txo[rxi]
                    n_passes=len(rxo)

                    for pass_i in range(n_passes):

                        po=rxo[pass_i]
                        print(po.keys())
                        #                        rfun=sint.interp1d(po["t"],po["range"])
                        rfun=self.interp_funs(po["t"],po["range"])
#                        rfun=sint.interp1d(po["t"],po["range"],kind="cubic")
                        snrfun=sint.interp1d(po["t"],po["snr"])
                        rrfun=sint.interp1d(po["t"],po["range_rate"])
                        rrdt=n.diff(po["t"])[0]
                        rrrfun=sint.interp1d(po["t"],n.gradient(po["range_rate"],rrdt))
                        #
                        passes.append({"t0":n.min(po["t"]),"t1":n.max(po["t"]),"oid":oi,"rxi":rxi,"po":po,"phase":0.0,"rfun":rfun,"rrfun":rrfun,"rrrfun":rrrfun,"snr":snrfun})

        print("n_objects %d n_rx %d"%(n_objects,n_rx))

        for p in passes:
            print("oid %d rxi %d t0 %1.2f t1 %1.2f"%(p["oid"],p["rxi"],p["t0"],p["t1"]))

        # go for it!
        dwos=[]
        arrs=[]

        # samples since 1970 of first data sample
        i0=int(n.min(t)*self.sr)

        # create tx channel
        tx_chdir="%s/tx"%(dirname)
        os.system("rm -rf %s"%(tx_chdir))
        os.system("mkdir -p %s"%(tx_chdir))

        # digital rf write object
        tx_dwo = drf.DigitalRFWriter(tx_chdir,          # directory
                                     n.complex64,       # dtype
                                     3600,              # subdir_cadence_secs
                                     1000,              # file_cadence_millisecs
                                     i0,                # start_global_index
                                     self.sr,           # sample_rate_numerator
                                     1,                 # sample_rate_denominator
                                     "fake_uuid",       # uuid_str
                                     0,                 # compression_level
                                     False,             # checksum
                                     True,              # is_complex
                                     1,                 # num_subchannels
                                     True,              # is_continuous
                                     True)              # marching_periods
        txz = n.zeros(self.ipp,dtype=n.complex64)

        for rxi in range(n_rx):
            chdir="%s/ch%03d"%(dirname,rxi)
            os.system("rm -rf %s"%(chdir))
            os.system("mkdir -p %s"%(chdir))
            dwo = drf.DigitalRFWriter(chdir,              # directory
                                      n.complex64,        # dtype
                                      3600,               # subdir_cadence_secs
                                      1000,               # file_cadence_millisecs
                                      i0,                 # start_global_index
                                      self.sr,            # sample_rate_numerator
                                      1,                  # sample_rate_denominator
                                      "fake_uuid",        # uuid_str
                                      0,                  # compression_level
                                      False,              # checksum
                                      True,               # is_complex
                                      1,                  # num_subchannels
                                      True,               # is_continuous
                                      True)               # marching_periods
            arr = n.zeros(self.ipp,dtype=n.complex64)
            dwos.append(dwo)
            arrs.append(arr)

        n_passes=len(passes)

        # time vector
        # self.ipp in the number of samples in an ipp
        tipp=n.arange(self.ipp+1,dtype=n.float64)/float(self.sr)

        print("n_rx %d"%(n_rx))
        print("generating tx")
        for i in range(n_ipp):

            tnow = float(i)*float(self.ipp)/float(self.sr)

            txz[:]=0.0
            for bi in range(self.n_bits):
                if self.code:
                    txz[(bi*self.bit_len):(bi*self.bit_len+self.bit_len)]=n.complex64(n.sign(n.random.randn(1)))
                else:
                    txz[(bi*self.bit_len):(bi*self.bit_len+self.bit_len)]=1.0

            tx_dwo.rf_write(txz)
        tx_dwo.close()

        txd = drf.DigitalRFReader(dirname)
        txb=txd.get_bounds("tx")
        print(txb)

        print("Generating RXs %d n_rx %d"%(n_passes,n_rx))

        for i in range(n_ipp):
            inow = i0+i*self.ipp
            tnow = inow/float(self.sr)

            # zero echo
            for ci in range(n_rx):
                arrs[ci][:]=0.0

            for pi in range(n_passes):
                po=passes[pi]
                if tnow > po["t0"] and tnow < (po["t1"]-self.ipp/self.sr):

                    rxi=po["rxi"]

                    rdelta = po["rfun"].rfun(tnow+tipp)

                    rdelta=rdelta-rdelta[0]
                    phase = n.mod(2.0*n.pi*rdelta/self.wavelength,2*n.pi)
                    csin=n.exp(1j*phase)*n.exp(1j*po["phase"])
                    po["phase"]=n.angle(csin[-1])

                    # total propagation range in samples
                    rs = int(n.round((po["rfun"].rfun(tnow)/c.c)*self.sr))

                    print("pass visible! t=%1.2f rs %d rxi %d oid %d r %1.2f rr %1.2f rrr %1.2f"%(tnow,rs,po["rxi"],po["oid"],po["rfun"].rfun(tnow),po["rrfun"](tnow),po["rrrfun"](tnow)))

                    # total range delayed index for start of transmit vector
                    txi0 = inow - rs

                    if txi0 > txb[0] and (txi0+self.ipp) < txb[1]:
                        txz = txd.read_vector_c81d(txi0,self.ipp,"tx")
                        sim_echo = csin[0:self.ipp]*txz
                        arrs[rxi][:]+=sim_echo

                        if False:
                            plt.plot(sim_echo.real)
                            plt.plot(sim_echo.imag)
                            plt.show()

            for ci in range(n_rx):
                dwo=dwos[ci]
                z=arrs[ci]
                dwo.rf_write(z)

        tx_dwo.close()
        for rxi in range(n_rx):
            dwos[rxi].close()

class Simulator(Config):
    """
    Simulate some raw voltage
    """
    @classmethod
    def get_default(cls):
        return {
            'sim_opts': {
                'r0' : 1000e3,
                'v0' : 2e3,
                'a0' : 80.0,
                'ipp' : 10000,
                'tx_len' : 2000,
                'bit_len' : 100,
                'n_ipp' : 100,
                'freq' : 230e6,
                'sr' : 1000000,
                'snr' : 30,
                }
            }

    def _set_values(self):
        self.dirname=self['sim_opts']['dirname']
        self.r0=float(self['sim_opts']['r0'])
        self.v0=float(self['sim_opts']['v0'])
        self.a0=float(self['sim_opts']['a0'])
        self.sr=int(self['sim_opts']['sr'])
        self.ipp=int(self['sim_opts']['ipp'])
        self.tx_len=int(self['sim_opts']['tx_len'])
        self.n_ipp=int(self['sim_opts']['n_ipp'])

        self.tx_len=int(self['sim_opts']['tx_len'])
        self.bit_len=int(self['sim_opts']['bit_len'])
        self.snr=float(self['sim_opts']['snr'])
        self.wavelength =c.c/float(self['sim_opts']['freq'])
        self.n_bits=int(self.tx_len / self.bit_len) #self['int(tx_len/bit_len)']

    def __init__(self, dir):
        super().__init__(dir)
        self._set_values()

    def run(self):
        # create tx channel
        tx_chdir="%s/tx"%(self.dirname)
        os.system("rm -rf %s"%(tx_chdir))
        os.system("mkdir -p %s"%(tx_chdir))

        i0=0

        # digital rf write object
        tx_dwo = drf.DigitalRFWriter(tx_chdir,          # directory
                                     n.complex64,       # dtype
                                     3600,              # subdir_cadence_secs
                                     100,               # file_cadence_millisecs
                                     i0,                # start_global_index
                                     self.sr,           # sample_rate_numerator
                                     1,                 # sample_rate_denominator
                                     "fake_uuid",       # uuid_str
                                     0,                 # compression_level
                                     False,             # checksum
                                     True,              # is_complex
                                     1,                 # num_subchannels
                                     True,              # is_continuous
                                     True)              # marching_periods

        txz = n.zeros(self.ipp,dtype=n.complex64)

        rxi=0
        chdir="%s/ch%03d"%(self.dirname,rxi)
        os.system("rm -rf %s"%(chdir))
        os.system("mkdir -p %s"%(chdir))

        dwo = drf.DigitalRFWriter(chdir,              # directory
                                  n.complex64,        # dtype
                                  3600,               # subdir_cadence_secs
                                  100,                # file_cadence_millisecs
                                  i0,                 # start_global_index
                                  self.sr,            # sample_rate_numerator
                                  1,                  # sample_rate_denominator
                                  "fake_uuid",        # uuid_str
                                  0,                  # compression_level
                                  False,              # checksum
                                  True,               # is_complex
                                      1,              # num_subchannels
                                  True,               # is_continuous
                                  True)               # marching_periods

        rxz = n.zeros(self.ipp,dtype=n.complex64)

        tipp=n.arange(self.ipp+1,dtype=n.float64)/float(self.sr)

        tx_pwr = 1.0

        # one sample noise power
        noise_pwr = tx_pwr/(10**(self.snr/10.0))

        ph0=0.0
        for i in range(self.n_ipp):
            tnow=i*self.ipp/float(self.sr)

            txz[:]=0.0
            for bi in range(self.n_bits):
                # random phase code
                txz[(bi*self.bit_len):(bi*self.bit_len+self.bit_len)]=n.complex64(n.sign(n.random.randn(1)))

            tx_dwo.rf_write(n.array(txz,dtype=n.complex64))

            rvec = self.r0 + self.v0*(tipp+tnow) + 0.5*self.a0*(tipp+tnow)**2.0
            rg = int(self.sr*rvec[0]/c.c)

#            print("%d %d %1.2f %1.2f"%(i,rg,tnow,self.v0+self.a0*(tipp[0]+tnow)))
            phase=n.fmod(2.0*n.pi*rvec/self.wavelength,2.0*n.pi)
            echo=n.roll(txz,rg)*n.exp(1j*phase[0:self.ipp])

            noise = n.array(n.random.randn(self.ipp) + 1j*n.random.randn(self.ipp),dtype=n.complex64)*(n.sqrt(noise_pwr)/n.sqrt(2.0))
            rxz[:]=echo + noise

            dwo.rf_write(n.array(rxz,dtype=n.complex64))



if __name__ == "__main__":
    dir = {
        'sim_opts' : {
            'dirname': '/tmp/test',
        }
    }
    sim = Simulator.from_dict(dir, from_default=True)
    sim.run()
    exit(0)

    # pulse length, number pulses integrated coherently,
    d=pickle.load(open("data/e3d_htpl_end_to_end/tracking.pickle","rb"))

    # raw voltage simulator class
    rs=raw_sim()
    #
    rs.raw_voltage(d)


