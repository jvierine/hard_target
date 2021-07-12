import numpy as n
import scipy.constants as sc
import os

try:
    import configparser
except ImportError as e:
    import configparser2 as configparser
    
import json
import os 

# configuration of hard target analysis
#
# doppler sign convention: range reduces with positive doppler velocity.
# 
class gmf_opts:
    
    def __str__(self):
        out="Configuration\n"
        for e in dir(self):
            if not callable(getattr(self,e)) and not e.startswith("__"):
                out+="%s = %s\n"%(e,getattr(self,e))
        return(out)

    def set_n_ranges(self, range_gate_0, n_range_gates):
        """
        Reset the number of range-gates. This is useful when reanalyzing with better resolution
        to fine-tune the result
        """
        self.n_range_gates=n_range_gates
        self.range_gate_0=range_gate_0

        # range gates to search through
        self.rgs=n.arange(self.n_range_gates)*self.range_gate_step+self.range_gate_0
        self.rgs_float=n.array(self.rgs,dtype=n.float32)
        
        # total propagation range
        self.ranges=self.rgs*sc.c/1e3/self.sample_rate

    def __init__(self,fname=None):
        c=configparser.ConfigParser()
        # initialize with default values
        c["config"]={"n_ipp":'5',
                     "data_dirs":'["/data0/2020.10.15/test1e6_4.04e6","/data1/2020.10.15/test1e6_4.04e6"]', # Data options, should not have default
                     "sample_rate":'1000000',
                     "n_range_gates":'10000',
                     "range_gate_0":'200',
                     "range_gate_step":'1',
                     "frequency_decimation":'25',
                     "ipp":'10000',                   # microseconds 
                     "tx_pulse_length":'445',  
                     "tx_bit_length":'20',
                     "ground_clutter_length":'1500',
                     "min_acceleration":'-400.0',
                     "max_acceleration":'400.0',
                     "acceleration_resolution":"0.2",
                     "snr_thresh":"10.0",                     
                     "save_parameters":'true',
                     "doppler_sign":'1.0',
                     "rx_channel":'"ch"', # Data options, should not have default
                     "tx_channel":'"ch"', # Data options, should not have default
                     "radar_frequency":'500e6',
                     "reanalyze":"true",
                     "output_dir":'"./spade_det"', # Data options, should not have default
                     "debug_plot":'true',
                     "debug_plot_acc":'true',
                     "debug_print":'true',
                     "round_trip_range":'true',
                     "num_cohints_per_file":'100',
                     "use_gpu":'false',
                     "use_python":'false',
                     "use_cpu":'true'}

        if fname != None:
            if os.path.exists(fname):
                print("reading configuration from %s"%(fname))
                c.read(fname)
            else:
                print("configuration file %s doesn't exist. using default values"%(fname))

        print(c.sections())
        print(c.keys())
        print(fname)
        self.fname=fname

        self.t0=None
        self.t1=None

        self.n_ipp=int(json.loads(c["config"]["n_ipp"]))
        self.data_dirs=json.loads(c["config"]["data_dirs"])
        print(self.data_dirs)
        self.sample_rate=float(json.loads(c["config"]["sample_rate"]))
        self.n_range_gates=int(json.loads(c["config"]["n_range_gates"]))
        self.range_gate_0=int(json.loads(c["config"]["range_gate_0"]))
        self.range_gate_step=int(json.loads(c["config"]["range_gate_step"]))
        self.frequency_decimation=int(json.loads(c["config"]["frequency_decimation"]))
        self.ipp=int(json.loads(c["config"]["ipp"]))
        self.tx_pulse_length=int(json.loads(c["config"]["tx_pulse_length"]))
        self.ground_clutter_length=int(json.loads(c["config"]["ground_clutter_length"]))
        self.min_acceleration=float(json.loads(c["config"]["min_acceleration"]))
        self.max_acceleration=float(json.loads(c["config"]["max_acceleration"]))
        self.acceleration_resolution=float(json.loads(c["config"]["acceleration_resolution"]))
        self.snr_thresh=float(json.loads(c["config"]["snr_thresh"]))
        self.save_parameters=bool(json.loads(c["config"]["save_parameters"]))
        self.doppler_sign=float(json.loads(c["config"]["doppler_sign"]))
        self.rx_channel=json.loads(c["config"]["rx_channel"])
        self.tx_channel=json.loads(c["config"]["tx_channel"])
        self.radar_frequency=float(json.loads(c["config"]["radar_frequency"]))
        self.output_dir=json.loads(c["config"]["output_dir"])
        self.debug_plot=bool(json.loads(c["config"]["debug_plot"]))
        self.debug_plot_acc=bool(json.loads(c["config"]["debug_plot_acc"]))
        self.debug_print=bool(json.loads(c["config"]["debug_print"]))
        self.debug_plot_data_read=False
        self.num_cohints_per_file=int(json.loads(c["config"]["num_cohints_per_file"]))
        self.use_gpu=bool(json.loads(c["config"]["use_gpu"]))
        self.use_python=bool(json.loads(c["config"]["use_python"]))
        self.reanalyze=bool(json.loads(c["config"]["reanalyze"]))        
        self.round_trip_range=bool(json.loads(c["config"]["round_trip_range"]))
        self.debug_gmf_output=True


        if self.save_parameters:
            os.system("mkdir -p %s"%(self.output_dir))
            print("mkdir -p %s"%(self.output_dir))
        
        # length of coherent integration
        self.n_fft=self.n_ipp*self.ipp

        # frequency vector 
        self.fvec=n.fft.fftfreq(int(self.n_fft/self.frequency_decimation),d=self.frequency_decimation/self.sample_rate)
        
        self.set_n_ranges(self.range_gate_0, self.n_range_gates)
        
        # range-rate is doppler-shift in hertz multiplied with wavelength 
        self.wavelength = sc.c/self.radar_frequency
        self.range_rates=self.doppler_sign*self.wavelength*self.fvec
        
        # time vector 
        times=self.frequency_decimation*n.arange(int(self.n_fft/self.frequency_decimation))/self.sample_rate
        times2=times**2.0

        # radar frequency in radians per second
        om=2.0*n.pi*self.radar_frequency

        # these are the accelerations we'll try out
        tau = self.n_ipp*self.ipp/self.sample_rate
        
        # acceleration sampled with 0.2 radian steps at the end of the coherent integration window
        delta_a = self.max_acceleration - self.min_acceleration
        self.n_accelerations = int(n.ceil( delta_a*(n.pi/self.wavelength)*tau**2.0 / self.acceleration_resolution))
        
        self.accs=n.linspace(self.min_acceleration,self.max_acceleration,num=self.n_accelerations) # m/s**2
        self.acc_phasors=n.zeros([self.n_accelerations,int(self.n_fft/self.frequency_decimation)],dtype=n.complex64)

        # precalculate phasors corresponding to different accelerations
        for ai,a in enumerate(self.accs):
            self.acc_phasors[ai,:]=n.exp(-1j*2.0*n.pi*(self.doppler_sign*0.5*self.accs[ai]/self.wavelength)*times2)
            
        # how many extra ipps do we need to read for coherent integration
        self.n_extra=int(n.ceil(n.max(self.rgs)/self.ipp))+1

        # this stencil is used to block tx pulses and ground clutter
        self.read_length=self.n_fft+self.n_extra*self.ipp
        self.rx_stencil=n.ones(self.read_length,dtype=n.float32)
        # this stencil is used to select tx pulses
        self.tx_stencil=n.ones(self.read_length,dtype=n.float32)    

        # for each coherently integrated IPP, create stencils
        for k in range(self.n_ipp+self.n_extra):
            self.tx_stencil[(k*self.ipp+self.tx_pulse_length):(k*self.ipp+self.ipp)]=0.0        
            # pad zeros to rx
            self.rx_stencil[(k*self.ipp):(k*self.ipp+self.tx_pulse_length+self.ground_clutter_length)]=0.0

        if self.debug_plot_acc:
            import matplotlib.pyplot as plt
            plt.plot(self.accs,self.acc_phasors.real[:,int(self.n_fft/self.frequency_decimation)-1])
            plt.plot(self.accs,self.acc_phasors.imag[:,int(self.n_fft/self.frequency_decimation)-1])
            plt.plot(self.accs,self.acc_phasors.real[:,int(self.n_fft/self.frequency_decimation)-1],"*")
            plt.plot(self.accs,self.acc_phasors.imag[:,int(self.n_fft/self.frequency_decimation)-1],"*")
            plt.xlabel("Accelerations (m/s^2)")
            plt.title("Acceleration phasors at maximum coherent integration")
            plt.show()
            # plot acceleration phasors
            plt.subplot(121)
            plt.pcolormesh(times/1e-3,self.accs,self.acc_phasors.real)
            plt.ylabel("Acceleration ($m/s^2$)")
            plt.xlabel("Time (ms)")
            plt.colorbar()
            plt.title("Acceleration phasors (Real component)")
            plt.subplot(122)
            plt.pcolormesh(times/1e-3,self.accs,self.acc_phasors.imag)
            plt.ylabel("Acceleration ($m/s^2$)")
            plt.xlabel("Time (ms)")
            plt.colorbar()
            plt.title("Acceleration phasors (Im component)")
            plt.show()

            plt.plot(n.arange(self.read_length),self.tx_stencil,label="tx stencil")
            plt.plot(n.arange(self.read_length),self.rx_stencil,label="rx stencil")
            plt.title("TX and RX stencils")
            plt.legend()
            plt.xlabel("Samples")
            plt.show()

if __name__ == "__main__":
    o=gmf_opts("cfg/esr_2018.ini")
    print(o)
