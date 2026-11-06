from .examples import run_sim
from . import __path__
import sys

print(__path__[0])

gpu = False
try:
    if sys.argv[1] == "gpu":
        gpu=True
        print("Running simulator using GPU optimized code")
    else:
        print("Running simulator using serial code")
except IndexError:
    gpu = False
    print("Running simulator using serial code")


run_sim.start_sim(gpu)
