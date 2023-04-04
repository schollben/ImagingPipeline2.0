#!/usr/bin/env Python
import os
import glob
import code
from motion_correction_script import register_one_session
from stacks_to_hdf5 import stacks_to_hdf5
from datetime import datetime

def register_bulk():
    sessions_to_run = [
        "D:\\BRUKER\\TSeries-03282023-1243-005"
    ]


    ### dataset dependent parameters
    fr = 30             # imaging rate in frames per second
    decay_time = 1 # Recommended by Ben on March 20th

 
    sparse_acq=False

    ### Based on Ben's recommendations
    dxy = (1.0, 1.0)
    max_shift_um = (32, 32)
    patch_motion_um = (64., 64.)
    max_shifts = [int(a/b) for a, b in zip(max_shift_um, dxy)]
    strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
    overlaps = (32, 32)
    max_deviation_rigid = 3
    #code.interact(local=locals())

    mc_dict = {
        'fr': fr,
        'decay_time': decay_time,
        'dxy': dxy,
        'pw_rigid': False,
        'max_shifts': max_shifts,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': 'copy',
        'nonneg_movie': False,
        'use_cuda': False,
        'niter_rig': 2
    }

    time_deltas = []
    for i in range(0,len(sessions_to_run)):
        print(sessions_to_run[i])
        t_start = datetime.now()
        stacks_to_hdf5(sessions_to_run[i], delete_tiffs=True)
        register_one_session(sessions_to_run[i], mc_dict, keep_memmap=False,
            save_sample=True, sample_name="01_rigid.h5")
        if sparse_acq:
            mc_dict['pw_rigid'] = True
            register_one_session(sessions_to_run[i], mc_dict, 
                keep_memmap=False, save_sample=True, sample_name="02_nonrigid.h5")
        t_stop = datetime.now()
        time_deltas.append(t_stop - t_start)
        print(str(time_deltas[i]))

if __name__ == "__main__":
    register_bulk()