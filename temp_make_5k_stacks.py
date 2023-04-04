#!/usr/env/bin

# Point to parent dir with giant HDF5 files, create 5k stacks

import h5py

parent_dir = "D:\\seagate_data_for_ben\\TSeries-08252022-1332-006\\Registered\\Channel1"

full_registered = h5py.File(parent_dir + "\\registered.h5", "r")
#full_denoised = h5py.File(parent_dir + "\\res_short_delme.h5","r")

short_registered =  h5py.File(parent_dir + "\\short_registered.h5", "w")
short_registered.create_dataset("mov", (5000, 512, 512))

#short_denoised =  h5py.File(parent_dir + "\\short_denoised.h5", "w")
#short_denoised.create_dataset("data", (5000, 512, 512))

short_registered["mov"][:,:,:] = full_registered["data"][0:5000, :, :]
#short_denoised["data"][:,:,:]   = full_denoised["data"][0:5000, :, :]

full_registered.close()
short_registered.close()
#hort_denoised.close()