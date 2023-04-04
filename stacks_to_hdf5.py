#!/usr/env/bin Python

# Convert tiff stacks into an HDF5 file for running through CaImAn
# Originally passed tiff stacks through, but because CaImAn doesn't
#	drop file handles for the tiff stacks once it's completed registration,
#	deletion becomes difficult.

import os
import glob
import h5py
import numpy as np
import caiman as cm

def stacks_to_hdf5(parent_dir, delete_tiffs):
    print("Converting from tiff stacks to HDF5 for " + str(parent_dir))
    
    # Cleaning up directories from previous run results.
    prev_samples_maybe = glob.glob(parent_dir + "\\0[1-3]*")
    for i in range(0, len(prev_samples_maybe)):
        os.remove(prev_samples_maybe[i])
    
    fnames = glob.glob(parent_dir + "\\*.tif")
    if len(fnames) > 0:
        
        prev_result = glob.glob(parent_dir + "\\*registered.h5")
        if len(prev_result) > 0:
            print("Removing previous HDF5 stack.")
            os.remove(prev_result[0])
        
        mov = cm.load(fnames[0])
        stack_size = mov.shape[0]
        
        # First and last 30 frames flipped and appended to movie since DeepInterpolation
        #   shaves off 30 frames of data on both ends when it runs.
        first_30 = np.flip(mov[0:30, :, :], 0)
        
        mov = cm.load(fnames[len(fnames) - 1])
        last_30 = np.flip(mov[mov.shape[0] - 30:, :, :], 0)
        
        num_frames = 60 + ((len(fnames) - 1) * stack_size) + mov.shape[0]

        f_out = h5py.File(parent_dir + "\\unregistered.h5", "w")
        f_out.create_dataset("mov", (num_frames, mov.shape[1], mov.shape[2]))

        f_out["mov"][0:30,:,:] = first_30
        frames_written = 30
        
        for i in range(0, len(fnames)):
            mov = cm.load(fnames[i])
            f_out["mov"][frames_written:frames_written+mov.shape[0], :, :] = np.array(mov)
            frames_written += mov.shape[0]
            del mov
            if delete_tiffs:
                os.remove(fnames[i])
            
        # Last 30
        f_out["mov"][frames_written:frames_written+30, :, :] = last_30
        f_out.close()
    else:
        print("No TIFF files were found to be converted.")
    
if __name__ == "__main__":
    stacks_to_hdf5("D:\\seagate_data_for_ben\\TSeries-08252022-1332-002\\Registered\\Channel1", True)
