import hardtarget.analysis.gmf_opts as go
import digital_rf as drf
import numpy as n
import matplotlib.pyplot as plt
import sys

def analyze_pwr(conf):
    
    d=drf.DigitalRFReader(conf.data_dirs)
    b_rx=d.get_bounds(conf.rx_channel)
    b_tx=d.get_bounds(conf.tx_channel)

    n_ipps=(b_rx[1]-b_rx[0])/conf.ipp
    for i in range(n_ipps):
        z=drf.read_vector_c81d(i*conf.ipp,conf.ipp,conf.rx_channel)
        plt.plot(z.real)
        plt.plot(z.imag)
        plt.show()
        

 #   n_ints=int(n.floor((b[1]-b[0])/(conf.ipp*conf.n_ipp))/conf.num_cohints_per_file)
    
#    print("RX bounds %s-%s TX bounds %s-%s"%(stuffr.unix2datestr(b_rx[0]/conf.sample_rate)))

                                             


if __name__ == "__main__":
    if len(sys.argv) == 2:
        print(sys.argv[1])
        conf=go.gmf_opts.from_file(sys.argv[1])
    else:
        print("Provide configuration file as command line option")
        exit(0)
    analyze_pwr(conf)
