from logging import debug
from matplotlib.pyplot import cla
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
import os
from ..config import Config
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
class gmf_opts(Config):
    
    @classmethod
    def get_default(cls):
        #Set default paramaters in a dictionary
        #Return object based on default params
        print("Setting default values")
        return {
                'gmf_opts': 
                    {
                    "n_ipp":5,
                    "sample_rate":1000000,
                    "n_range_gates":10000,
                    "range_gate_0":200,
                    "range_gate_step":1,
                    "frequency_decimation":25,
                    "ipp":10000, 
                    "tx_pulse_length":445,  
                    "tx_bit_length":20,
                    "ground_clutter_length":1500,
                    "min_acceleration":-400.0,
                    "max_acceleration":400.0,
                    "acceleration_resolution":0.2,
                    "snr_thresh":10.0,                     
                    "save_parameters":True,
                    "doppler_sign":1.0,
                    "radar_frequency":500e6,
                    "reanalyze":True,
                    "debug_plot":False,
                    "debug_plot_acc":False,
                    "debug_print":False,
                    "round_trip_range":True,
                    "num_cohints_per_file":100,
                    "use_gpu":False,
                    "use_python":False,
                    "use_cpu":True,
                    },
                }

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
        self.rgs=np.arange(self.n_range_gates)*self.range_gate_step+self.range_gate_0
        self.rgs_float=np.array(self.rgs,dtype=np.float32)
        
        # total propagation range
        self.ranges=self.rgs*sc.c/1e3/self.sample_rate

    def set_values(self):
        self.t0=None
        self.t1=None
        
        #Non default paramaters
        try:
            self.data_dirs=self['gmf_opts']["data_dirs"]
            self.rx_channel=self['gmf_opts']["rx_channel"]
            self.tx_channel=self['gmf_opts']["tx_channel"]
            self.output_dir=self['gmf_opts']["output_dir"]
            self.n_ipp=int(self['gmf_opts']["n_ipp"])
            print(self.data_dirs)
            self.sample_rate=float(self['gmf_opts']["sample_rate"])
            self.n_range_gates=int(self['gmf_opts']["n_range_gates"])
            self.range_gate_0=int(self['gmf_opts']["range_gate_0"])
            self.range_gate_step=int(self['gmf_opts']["range_gate_step"])
            self.frequency_decimation=int(self['gmf_opts']["frequency_decimation"])
            self.ipp=int(self['gmf_opts']["ipp"])
            self.tx_pulse_length=int(self['gmf_opts']["tx_pulse_length"])
            self.ground_clutter_length=int(self['gmf_opts']["ground_clutter_length"])
            self.min_acceleration=float(self['gmf_opts']["min_acceleration"])
            self.max_acceleration=float(self['gmf_opts']["max_acceleration"])
            self.acceleration_resolution=float(self['gmf_opts']["acceleration_resolution"])
            self.snr_thresh=float(self['gmf_opts']["snr_thresh"])
            self.doppler_sign=float(self['gmf_opts']["doppler_sign"])
            self.radar_frequency=float(self['gmf_opts']["radar_frequency"])
            print(self['gmf_opts']["debug_plot_acc"])
            self.debug_plot_data_read=False
            self.num_cohints_per_file=int(self['gmf_opts']["num_cohints_per_file"])

            self.save_parameters=self._parse_bool(self['gmf_opts']["save_parameters"])
            self.debug_plot=self._parse_bool(self['gmf_opts']["debug_plot"])
            self.debug_plot_acc=self._parse_bool(self['gmf_opts']["debug_plot_acc"])
            self.debug_print=self._parse_bool(self['gmf_opts']["debug_print"])
            self.use_gpu=self._parse_bool(self['gmf_opts']["use_gpu"])
            self.use_python=self._parse_bool(self['gmf_opts']["use_python"])
            self.reanalyze=self._parse_bool(self['gmf_opts']["reanalyze"])       
            self.round_trip_range=self._parse_bool(self['gmf_opts']["round_trip_range"])
            self.debug_gmf_output=True
        except KeyError as e:
            #Raise valuerror instead of KeyError if some values are missing
            raise ValueError("Missing Value: " + str(e))

    def _parse_bool(self, string) -> bool:
        """
        Method to parse string into boolean

        Paramaters:
            string: Any type.
        
        Returns:
            Boolean
        """
        if isinstance(string, bool):
            return string
        elif string.lower() in ['true', 't']:
            return True
        elif string.lower() in ["false", 'f']:
            return False
        else:
            raise ValueError(f"Unknown boolean operator: {string}")

    def __init__(self,paramaters, values_as_strings = False):

        super().__init__(paramaters,values_as_strings)

        self.set_values()
        if self.save_parameters:
            self.save_param('config',output_dir=self.output_dir, ini=self.values_as_strings)
        
        # length of coherent integration
        self.n_fft=self.n_ipp*self.ipp

        # frequency vector 
        self.fvec=np.fft.fftfreq(int(self.n_fft/self.frequency_decimation),d=self.frequency_decimation/self.sample_rate)
        
        self.set_n_ranges(self.range_gate_0, self.n_range_gates)
        
        # range-rate is doppler-shift in hertz multiplied with wavelength 
        self.wavelength = sc.c/self.radar_frequency
        self.range_rates=self.doppler_sign*self.wavelength*self.fvec
        
        #Time vector
        times=self.frequency_decimation*np.arange(int(self.n_fft/self.frequency_decimation))/self.sample_rate
        times2=times**2.0

        # radar frequency in radians per second
        om=2.0*np.pi*self.radar_frequency

        # these are the accelerations we'll try out
        tau = self.n_ipp*self.ipp/self.sample_rate
        
        # acceleration sampled with 0.2 radian steps at the end of the coherent integration window
        delta_a = self.max_acceleration - self.min_acceleration
        self.n_accelerations = int(np.ceil( delta_a*(np.pi/self.wavelength)*tau**2.0 / self.acceleration_resolution))
        
        self.accs=np.linspace(self.min_acceleration,self.max_acceleration,num=self.n_accelerations) # m/s**2
        self.acc_phasors=np.zeros([self.n_accelerations,int(self.n_fft/self.frequency_decimation)],dtype=np.complex64)

        # precalculate phasors corresponding to different accelerations
        for ai,a in enumerate(self.accs):
            self.acc_phasors[ai,:]=np.exp(-1j*2.0*np.pi*(self.doppler_sign*0.5*self.accs[ai]/self.wavelength)*times2)
            
        # how many extra ipps do we need to read for coherent integration
        self.n_extra=int(np.ceil(np.max(self.rgs)/self.ipp))+1

        # this stencil is used to block tx pulses and ground clutter
        self.read_length=self.n_fft+self.n_extra*self.ipp
        self.rx_stencil=np.ones(self.read_length,dtype=np.float32)
        # this stencil is used to select tx pulses
        self.tx_stencil=np.ones(self.read_length,dtype=np.float32)    

        # for each coherently integrated IPP, create stencils
        for k in range(self.n_ipp+self.n_extra):
            self.tx_stencil[(k*self.ipp+self.tx_pulse_length):(k*self.ipp+self.ipp)]=0.0        
            # pad zeros to rx
            self.rx_stencil[(k*self.ipp):(k*self.ipp+self.tx_pulse_length+self.ground_clutter_length)]=0.0
        
        if self.debug_plot_acc:
            self.plot_debug(show=True)
    
    def plot_debug(self, show=False, save=False):
        # time vector 
        times=self.frequency_decimation*np.arange(int(self.n_fft/self.frequency_decimation))/self.sample_rate
        times2=times**2.0

        if save:
            os.system("mkdir -p %s/debug_plots"%(self.output_dir))
            path = self.output_dir + '/debug_plots/'

        plt.plot(self.accs,self.acc_phasors.real[:,int(self.n_fft/self.frequency_decimation)-1])
        plt.plot(self.accs,self.acc_phasors.imag[:,int(self.n_fft/self.frequency_decimation)-1])
        plt.plot(self.accs,self.acc_phasors.real[:,int(self.n_fft/self.frequency_decimation)-1],"*")
        plt.plot(self.accs,self.acc_phasors.imag[:,int(self.n_fft/self.frequency_decimation)-1],"*")
        plt.xlabel("Accelerations (m/s^2)")
        plt.title("Acceleration phasors at maximum coherent integration")

        if save:
            plt.savefig(path + '/acc_phasor_01.pdf')
        if show:
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

        if save:
            plt.savefig(path + '/acc_phasor_02.pdf')
        if show:
            plt.show()

        plt.plot(np.arange(self.read_length),self.tx_stencil,label="tx stencil")
        plt.plot(np.arange(self.read_length),self.rx_stencil,label="rx stencil")
        plt.title("TX and RX stencils")
        plt.legend()
        plt.xlabel("Samples")
        
        if save:
            plt.savefig(path + '/stencils.pdf')
        if show:
            plt.show()

if __name__ == "__main__":
    dirs = {
        "gmf_opts": {
            "data_dirs": '/tmp',
            "rx_channel": '/tmp',
            "tx_channel": '/tmp',
            "output_dir": '/tmp',
        },
    }
    # o=gmf_opts.from_dict(dirs, True)
    o=gmf_opts.from_file('./cfg/sim.ini', True)
    print(o)
