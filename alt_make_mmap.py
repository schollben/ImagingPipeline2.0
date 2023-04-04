#!/usr/env/bin Python
import glob
import h5py
import math
import numpy as np

# Takes the massive HDF5 files out of DeepInterpolation and creates
#   NumPy memory mapped arrays in the same parent directory.
# CaImAn's memory map scripts presume all data fits in memory during typecasting
#   and will fail on any reasonably large datasets.

def make_mmmap(parent_dir):
    f = h5py.File(parent_dir + "\\res.h5", "r")
    numframes = f["data"].shape[0]

    chunk_size = 5000
    numwrites = math.floor(numframes / chunk_size)
    frames_written = 0

    out_name = "d1_" + f["data"].shape[1] + "_d2_"+f["data"].shape[2]+ "_d3_1_order_C_frames_"+numframes

    big_mov = np.memmap(parent_dir + "\\big_mov", mode="w+", 
            dtype=np.float32, shape=(numframes, 512, 512), order='C')
    del big_mov
        
    holder = np.zeros([chunk_size, 512, 512])   
    
    for i in range(0,numwrites):
        print(str(i*chunk_size)+" to "+str((i+1)*chunk_size))
        f_out = h5py.File(parent_dir + "\\" + str(i) + ".h5py", "w")
        f_out.create_dataset("data", (chunk_size, 512, 512))
   
        big_mov = np.memmap(parent_dir + "\\big_mov", mode="r+", 
            dtype=np.float32, shape=(numframes, 512, 512), order='C')
        holder[:,:,:] = f["data"][i*chunk_size:(i+1)*chunk_size, :, :]
        big_mov[i*chunk_size:(i+1)*chunk_size,:, :] = holder
   
        f_out["data"][:,:,:] = holder
        f_out.close()
   
        frames_written += chunk_size
    # Last bit
    big_mov = np.memmap(parent_dir + "\\big_mov", mode="r+", 
        dtype=np.float32, shape=(numframes, 512, 512), order='C')

    holder = f["data"][frames_written:numframes, :, :]
    big_mov[frames_written:numframes,:, :] = holder



if __name__ == '__main__':
    make_mmap(parent_dir)