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
        e = 0.1, 
        i = 75, 
        raan = 79,
        aop = 0,
        mu0 = mu0,
        epoch = 53005.0,
        parameters = dict(
            d = 0.05,
        ),
        oid = oid,
        logger = logger,
    )
    for oid, mu0 in enumerate([62.0, 61.8])
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

output = observe_objects(
    objs, 
    controllers, 
    end_t = ps.end(), 
    verbose = True,
)

with open('./incidental_tracking.pickle', 'wb') as h:
    pickle.dump(output, h)

#structure is output['observations'][object][tx id][rx id][pass number]
for oid in range(len(objs)):
    sorts.plotting.observed_parameters(output['observations'][oid][0][0])
plt.show()
