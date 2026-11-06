# Stuff that is useful in several places in hardtarget library

import digital_rf as drf

# Juha's convenience function for reading 1d vector of c8 values from a digital_rf source
def read_vector_c81d(d, *args, **kws):
    return d.read_vector_1d(*args, **kws).astype('c8', casting='unsafe', copy=False)

