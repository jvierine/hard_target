#!/usr/bin/env python
import pickle

import numpy as np
import matplotlib.pyplot as plt

import sorts

from observe import logger, observe_objects 

scan = sorts.radar.scans.Fence(azimuth=90, num=40, dwell=0.1, min_elevation=30)

end_t = 600.0

objs = [
    sorts.SpaceObject(
        sorts.propagator.SGP4,
        propagator_options = dict(
            settings = dict(
                in_frame='TEME',
                out_frame='ITRF',
            ),
        ),
        a = 7200e3, 
        e = 0.02, 
        i = 75, 
        raan = 86,
        aop = 0,
        mu0 = 60,
        epoch = 53005.0,
        parameters = dict(
            d = 0.1,
        ),
        logger = logger,
    ),
]

t_slices = np.arange(0, end_t, scan.dwell())
controllers = [sorts.controller.Scanner(None, scan, t = t_slices)]

output = observe_objects(
    objs, 
    controllers, 
    end_t = end_t, 
    verbose = True,
)

with open('./scanning.pickle', 'wb') as h:
    pickle.dump(output, h)

#structure is output['observations'][object][tx id][rx id][pass number]
sorts.plotting.observed_parameters(output['observations'][0][0][0])
plt.show()