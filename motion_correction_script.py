#!/usr/bin/env python
import os
import os.path
import cv2
import math
import h5py
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np


import code
from time import process_time
    
try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass


import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.summary_images import local_correlations_movie_offline


def register_one_session(parent_dir, mc_dict, keep_memmap, save_sample, sample_name):
    #pass  # For compatibility between running under Spyder and the CLI
    
    fnames = glob.glob(parent_dir + "\\*registered.h5")
    prev_mmap = glob.glob(parent_dir + "\\*mmap")
    
    assert len(fnames) > 0 or len(prev_mmap) > 0, "No file detected for " + parent_dir + "\n Halting."
    assert len(fnames) < 2 or len(prev_mmap) > 0, "More than one file detected for " + parent_dir + "\n Halting."
    
    if len(prev_mmap) > 0:
        print("Using mmap from previous run.")
        fnames = prev_mmap

    mc_dict['fnames'] = fnames
    mc_dict['upsample_factor_grid'] = 8 # Attempting to fix subpixel registration issue

    opts = params.CNMFParams(params_dict=mc_dict)

# %% start a cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))

    mc.motion_correct(save_movie=True)

 # Storing results in HDF5 for DeepInterpolation
    if mc_dict['pw_rigid']:
        numframes = len(mc.x_shifts_els)
    else:
        numframes = len(mc.shifts_rig)

    os.replace(fnames[0], parent_dir + "\\registered.h5")
    datafile = h5py.File(parent_dir + '\\registered.h5', 'w')
    datafile.create_dataset("mov", (numframes, 512, 512))
    
    fnames_new = glob.glob(parent_dir + "\\*.mmap")
    frames_written = 0
    
    for i in range(0, math.floor(numframes / 1000)):
        mov = cm.load(fnames_new[0])
        temp_data = np.array(mov[frames_written:frames_written + 1000, :, :])
        datafile["mov"][frames_written:frames_written + 1000, :, :] = temp_data
        frames_written += 1000
        del mov
    
    # handling last point
    if numframes > frames_written:
        mov = cm.load(fnames_new[0])
        temp_data = np.array(mov[frames_written:mov.shape[0], :, :])
        datafile["mov"][frames_written:mov.shape[0], :, :] = temp_data
        del mov
        
    del temp_data        
    cm.stop_server(dview=dview)

    if not keep_memmap:
        for i in range(0, len(fnames_new)):
            os.remove(fnames_new[i])

    if save_sample:
        # create directory for samples
        sample_size = min(2000, numframes)
        if not os.path.isdir(parent_dir + "\\samples"):
            os.makedirs(parent_dir + "\\samples")
        samplefile = h5py.File(parent_dir + "\\samples\\" + sample_name, "w")
        samplefile.create_dataset("mov", (sample_size, 512, 512))
        samplefile["mov"][:,:,:] = datafile["mov"][0:sample_size,:,:]
        samplefile.close()
        
    datafile.close()

if __name__ == "__main__":
    start_time = process_time()
    main()
    stop_time = process_time()
    print("Elapsed: ", stop_time, start_time)