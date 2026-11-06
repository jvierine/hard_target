#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


import sorts
eiscat3d = sorts.radars.eiscat3d_interp

from sorts.scheduler import StaticList, ObservedParameters
from sorts.controller import Scanner
from sorts import SpaceObject
from sorts.profiling import Profiler
from sorts.radar.scans import Fence

logger = sorts.profiling.get_logger('observe')

def observe_objects(objs, controllers, end_t=600.0, dt=10.0, verbose=True):

    p = Profiler()

    class ObservedScanning(StaticList, ObservedParameters):
        pass

    #set radar
    for ctrl in controllers:
        ctrl.radar = eiscat3d
        ctrl.profiler = p
        ctrl.logger = logger

    p.start('total')
    scheduler = ObservedScanning(
        radar = eiscat3d, 
        controllers = controllers, 
        logger = logger,
        profiler = p,
    )

    t = np.arange(0, end_t, dt)

    if verbose:
        for obj in objs: print(obj)

    datas = []
    passes = []
    states = []
    for ind in range(len(objs)):
        
        p.start('get_state')
        states += [objs[ind].get_state(t)]
        p.stop('get_state')

        p.start('find_passes')
        passes += [eiscat3d.find_passes(t, states[ind], cache_data = True)] 
        p.stop('find_passes')

        p.start('observe_passes')
        data = scheduler.observe_passes(passes[ind], space_object = objs[ind], snr_limit=False)
        p.stop('observe_passes')

        datas.append(data)

    p.stop('total')
    if verbose:
        print(p.fmt(normalize='total'))

    return {'t': t, 'observations': datas, 'passes': passes, 'states': states}
