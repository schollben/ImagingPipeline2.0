#!/usr/bin/env python
import cv2
import glob
import h5py
import math
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

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

# %%
# Set up the logger (optional); change this if you like.
# You can log to a file using the filename parameter, or make the output more
# or less verbose by setting level to logging.DEBUG, logging.INFO,
# logging.WARNING, or logging.ERROR

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]" \
                    "[%(process)d] %(message)s",
                    level=logging.WARNING)
def main():
    parent_dir = "D:\\pipeline_testing\\presentation\\TSeries-11052022-0005-005_and_0008-008\\Registered\\Channel1"
    f = h5py.File(parent_dir + "\\res.h5", "r")
    #numframes = f["data"].shape[0]
    numframes = 8000
    chunk_size = 4000
    numwrites = math.floor(numframes / chunk_size)
    frames_written = 0

    out_name = "memmap_d1_" + str(f["data"].shape[1]) + "_d2_"+ str(f["data"].shape[2]) + "_d3_1_order_C_frames_"+str(numframes) + "_.mmap"

    big_mov = np.memmap(parent_dir + "\\" + out_name, mode="w+", 
            dtype=np.float32, shape=(numframes, 512, 512), order='C')
    del big_mov
        
    holder = np.zeros([chunk_size, 512, 512])   
    
    for i in range(0,numwrites):
        print(str(i*chunk_size)+" to "+str((i+1)*chunk_size))
        f_out = h5py.File(parent_dir + "\\" + str(i) + ".h5py", "w")
        f_out.create_dataset("data", (chunk_size, 512, 512))
   
        big_mov = np.memmap(parent_dir + "\\" + out_name, mode="r+", 
            dtype=np.float32, shape=(numframes, 512, 512), order='C')
        holder[:,:,:] = f["data"][i*chunk_size:(i+1)*chunk_size, :, :]
        big_mov[i*chunk_size:(i+1)*chunk_size,:, :] = holder
   
        f_out["data"][:,:,:] = holder
        f_out.close()
   
        frames_written += chunk_size
    # Last bit
    print(str(frames_written) + " to " + str(numframes))
    big_mov = np.memmap(parent_dir + "\\" + out_name, mode="r+", 
        dtype=np.float32, shape=(numframes, 512, 512), order='C')

    holder = f["data"][frames_written:numframes, :, :]
    big_mov[frames_written:numframes,:, :] = holder
    del big_mov
    
    fnames_new = glob.glob(parent_dir + "\\*d1*d2*mmap")[0]
    Yr, dims, T = cm.load_memmap(fnames_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='C')
    # %% restart cluster to clean up memory
    # cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

    # Calculation to determine peak signal to noise

    # %%  parameters for source extraction and deconvolution
    p = 2  # order of the autoregressive system (2 for fast framerate, slow indicator)
    gnb = 1  # number of global background components (DeepInterpolation reduces global background complexity)
    merge_thr = 0.8  # merging threshold, max correlation allowed
    K = 10  # number of components per patch
    gSig = [7, 7]  # expected half size of neurons in pixels
    rf = max(10 * gSig[0], 10 * gSig[1])
    # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 4 * gSig[0]  # amount of overlap between the patches in pixels
    # initialization method (if analyzing dendritic data using 'sparse_nmf')
    method_init = 'greedy_roi'
    ssub = 2  # spatial subsampling during initialization
    tsub = 2  # temporal subsampling during intialization
    ring_size_factor = 1.7
    min_pnr = 20

    # others to add - need to change to loading previous opts from motion correction
    fr = 30
    mc_dict = {'fr': fr}

    opts = params.CNMFParams(params_dict=mc_dict)

    # parameters for component evaluation
    opts_dict = {'fnames': fnames_new,
                 'p': p,
                 'fr': fr,
                 'nb': gnb,
                 'rf': rf,
                 'K': K,
                 'gSig': gSig,
                 'stride': stride_cnmf,
                 'method_init': method_init,
                 'rolling_sum': True,
                 'merge_thr': merge_thr,
                 'n_processes': n_processes,
                 'only_init': True,
                 'ssub': ssub,
                 'tsub': tsub,
                 'seed_method': 'auto',
                 'ring_size_factor': ring_size_factor,
                 'min_pnr': min_pnr,
                 'p_patch': 1}

    opts.change_params(params_dict=opts_dict)
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)
    # %% plot contours of found components
    # Do this for just 1k frames. Don't grab the whole thing
    Cns = local_correlations_movie_offline(parent_dir + "\\000001.tif",
                                           remove_baseline=True, window=1000, stride=1000,
                                           winSize_baseline=100, quantil_min_baseline=50,
                                           dview=dview)
    #Cns = local_correlations_movie_offline(mc.mmap_file[0],
    #                                       remove_baseline=True, window=1000, stride=1000,
    #                                       winSize_baseline=100, quantil_min_baseline=10,
    #                                       dview=dview)
    Cn = Cns.max(axis=0)
    Cn[np.isnan(Cn)] = 0
    cnm.estimates.plot_contours(img=Cn)
    # plt.title('Contour plots of found components')
    # %% save results
    cnm.estimates.Cn = Cn
    #cnm.save(fname_new[:-5]+'_init.hdf5')
    #code.interact(local=locals())
    # %% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
    cnm2 = cnm.refit(images, dview=dview)
    # %% COMPONENT EVALUATION
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier
    min_SNR = 4  # signal to noise ratio for accepting a component
    rval_thr = 0.8  # space correlation threshold for accepting a component
    cnn_thr = 0.9  # threshold for CNN based classifier
    cnn_lowest = 0.30  # neurons with cnn probability lower than this value are rejected
    decay_time = 0.4
    cnm2.params.set('quality', {'decay_time': decay_time,
                                'min_SNR': min_SNR,
                                'rval_thr': rval_thr,
                                'use_cnn': True,
                                'min_cnn_thr': cnn_thr,
                                'cnn_lowest': cnn_lowest})

    cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
    cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)
    code.interact(local=locals())

    # Visual inspection, see a component that was rejected, but I know is a cell.
    #good_comp = cnm2.estimates.idx_components
    #bad_comp = cnm2.estimates.idx_components_bad

    #swap = np.array([7])

    #good_comp = np.append(good_comp, swap)
    #bad_comp = np.setdiff1d(bad_comp, swap)

    cnm2.estimates.select_components(use_object=True)
    #cnm2.estimates.select_components(good_comp)
        # Doing this actually clears the idx_components, idx_components_bad fields.
        # Have to regenerate these with evaluate_components after doing this.

    # %% Extract DF/F values
    cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)

    # %% Show final traces
    #cnm2.estimates.view_components(img=Cn)
    # %%
    cnm2.estimates.Cn = Cn
    #cnm2.save(cnm2.mmap_file[:-4] + 'hdf5')
    cnm2.save('TSeries-11052022_00001-10k_frames.hdf5')
    # also, cnm2 = cnmf.cnmf.load_CNMF('secondPass_secondAttempt.hdf5')

    code.interact(local=locals())
    cm.stop_server(dview=dview)

if __name__ == "__main__":
    start_time = process_time()
    main()
    stop_time = process_time()
    print("Elapsed: ", stop_time, start_time)