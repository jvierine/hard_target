# Hard target processing library (HTPL)

General package for hard target processing of radar data in the __digital\_rf__ format using matched filtering. Also includes raw data simulation for validation, testing and performance evaluation purposes.

## Install

This library is only intended for use on Unix systems.

The library can be installed through pip

```bash
pip install hardtarget
```

If there is an available cuda compiler in the environment, then both a CUDA library and a C library will be compiled. If no CUDA compiler is found, then only the C code will be compiled.



## Usage

The library operates on level 2 beamformed data streams that are stored on disk in Digital RF format. The program assumes that there is a complex baseband recording of the transmit waveform and the received complex voltage containing the radar echoes. 

The program can be adapted to various different radars and radar experiment parameters. This is done by editing a configuration file. An example configuration file is shown below. The example file includes comments that describe each parameter.

```bash
[config]
# data stream directory
data_dirs=["/media/j/ebd77b41-7efd-4238-b6f8-2b17bc33c84c/debsim"]
# how many range-gates are to be analyze
n_range_gates=2000
# how many samples do we remove due to ground clutter
ground_clutter_length=0
# minimum acceleration to search for (m/s^2)
min_acceleration=0.0
# maximum acceleration to search for (m/s^2)
max_acceleration=200.0
# how many radians of resolution do we want for the acceleration search
# (radians)
acceleration_resolution=0.2
# do we save analyzed results
save_parameters=true
# what is the Doppler shift sign convention
doppler_sign=1.0
# this channel contains the echoes
rx_channel="ch000"
# this channel contains the sampled transmit waveform
tx_channel="tx"
# what is the exact frequency of the radar
radar_frequency=230e6
# where will the results be stored
output_dir="./spade_det"
# display debug plots
debug_plot=false
debug_plot_acc=false
debug_print=false
# use round-trip range convention. if false, use total range and range-rate
round_trip_range=false
# reanalyze by overwriting already analyzed portions of the data
reanalyze=true
# how many coherent integration periods do we include in one output file
# smaller means that lower latency can be achieved
num_cohints_per_file=1
# use a GPU library to calculate the matched filter
# if false, use a CPU library written in C
use_gpu=false
# smallest signal-to-noise ratio to accept as target
snr_thresh=10.0
# inter-pulse period length in samples
ipp=100000
# transmit-pulse length in samples
tx_pulse_length=20000
# sample-rate in Hertz
sample_rate=10000000
# the first range gate to analyze (samples)
range_gate_0=1000
# how many samples do we step in range in the first stage of the course search
range_gate_step=50
# how much decimation in frequency do we perform (this affects minimum and maximum allowed Doppler shift)
frequency_decimation=100
# number of interpulse periods to coherently integrate
n_ipp=10
```

The program is divided into several sub-programs. The main program that performs the GMF analysis is `analyze_gmf.py`. This supports use of MPI for parallel execution. It is run together with the configuration file, which is a mandatory parameter:
```bash
python -m hardtarget.analyze_gmf config.ini
```
There is also an end-to-end test, which includes a raw voltage simulator that creates a dataset and analyzes it. This can be used to test the performance of the analysis program and to validate the results. This also includes use of an additional refinement step that is used to refine the detected targets:

To run the simulator using the compiled C code:
```bash
python -m hardtarget 
```

To run the simulator using the compiled CUDA code:
```bash
pythin -m hardtarget gpu
```

