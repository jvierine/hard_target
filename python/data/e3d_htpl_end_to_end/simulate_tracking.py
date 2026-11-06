#!/usr/bin/env python
import pickle

import numpy as np
import matplotlib.pyplot as plt

import sorts

from observe import logger, observe_objects, eiscat3d

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
        e = 0.01, 
        i = 75, 
        raan = 79,
        aop = 0,
        mu0 = 60,
        epoch = 53005.0,
        parameters = dict(
            d = 1.0,
        ),
        oid = 1,
        logger = logger,
    ),
]

t_find = np.arange(0, 3600*12.0, 10.0)
passes = eiscat3d.find_passes(t_find, objs[0].get_state(t_find), cache_data = False)

#track first pass
ps = passes[0][0][0]

#create tracklet points (can change later)
t = np.arange(ps.start(), ps.end(), 0.1)
states = objs[0].get_state(t)

#create the tracker
controllers = [sorts.controller.Tracker(radar=eiscat3d, t=t, ecefs=states[:3,], dwell = 0.1)]
controllers[0].meta['target'] = objs[0].oid

output = observe_objects(
    objs, 
    controllers, 
    end_t = ps.end(), 
    verbose = True,
)

with open('./tracking.pickle', 'wb') as h:
    pickle.dump(output, h)

#structure is output['observations'][object][tx id][rx id][pass number]
sorts.plotting.observed_parameters(output['observations'][0][0][0])
plt.show()