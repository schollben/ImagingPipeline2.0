import os
import sys
import glob
import h5py
import pathlib
from datetime import datetime

sys.path.append('C:\\Users\\greg\\Documents\\Python\\deepinterpolation-0.1.5')

from deepinterpolation.cli.fine_tuning import FineTuning

def main():
    loss_fun = "mean_squared_error"
    sessions_to_run = [
        "D:\\BRUKER\\TSeries-03282023-1243-005",
        #"D:\\seagate_data_for_ben\\TSeries-08252022-1332-003\\Registered\\Channel1",
    ]
    
    time_deltas = []
    for i in range(0, len(sessions_to_run)):
        parent_dir = sessions_to_run[i]
        finetuning_params = {}
        generator_param = {}
        generator_test_param = {}
    
        # Grab some info from datafile
        data_file_maybe = glob.glob(parent_dir + "\\registered.h5")
        assert len(data_file_maybe) == 1, "Either HDF5 file for registered data was not found, or more than one was found. Check parent directory."
        data_file = data_file_maybe[0]
    
    
        f = h5py.File(data_file, "a")
        rename = False
        try:
            total_frames = f['mov'].shape[0]
            rename = True
        except:
            total_frames = f['data'].shape[0]
    
        if total_frames < 14000:
            num_training_samples = total_frames
        else:
            num_training_samples = 14000
        
        if rename:
            f["data"] = f["mov"]
            del f["mov"]
        f.close()
    
        # Parameters used for the validation step each epoch
        generator_test_param["name"] = "OphysGenerator"
        generator_test_param["pre_frame"] = 30
        generator_test_param["post_frame"] = 30
        generator_test_param["data_path"] = data_file
        generator_test_param["batch_size"] = 10
        generator_test_param["start_frame"] = 1
        generator_test_param["end_frame"] = -1
        generator_test_param["total_samples"] = 500
        generator_test_param["randomize"] = 1
        generator_test_param["pre_post_omission"] = 0 
 
        #Parameters used for the main data generator
        generator_param["name"] = "OphysGenerator"
        generator_param["pre_frame"] = 30
        generator_param["post_frame"] = 30
        generator_param["data_path"] = data_file
        generator_param["batch_size"] = 10
        generator_param["start_frame"] = 1
        generator_param["end_frame"] = -1
        generator_param["total_samples"] = num_training_samples
        generator_param["randomize"] = 1
        generator_param["pre_post_omission"] = 0

        # Parameters used for the training process
        finetuning_params["name"] = "transfer_trainer"

        # Change this path to any model you wish to improve
        filename = "2019_09_11_23_32_unet_single_1024_mean_absolute_error_Ai93-0450.h5"
        local_path = \
            os.path.join(
                "C:\\Users\\greg\\Documents\\Python\\deepinterpolation-0.1.5\\sample_data",
                filename
            )
        finetuning_params["model_source"] = {
            "local_path": local_path
        }

        steps_per_epoch = 200
        finetuning_params["steps_per_epoch"] = steps_per_epoch
        finetuning_params[
            "period_save"
        ] = 5

        finetuning_params["learning_rate"] = 0.0001
        finetuning_params["loss"] = loss_fun
        finetuning_params["output_dir"] = parent_dir
        finetuning_params["apply_learning_decay"] = False
        finetuning_params["caching_validation"] = False
        finetuning_params["use_multiprocessing"] = False
        finetuning_params["nb_workers"] = 5

        args = {
            "finetuning_params": finetuning_params,
            "generator_params": generator_param,
            "test_generator_params": generator_test_param,
            "output_full_args": True
        }

        t_start = datetime.now()
        finetuning_obj = FineTuning(input_data=args, args=[])
        finetuning_obj.run()
        t_stop = datetime.now()
        time_deltas.append(t_stop - t_start)
        print(str(t_stop - t_start))

if __name__=='__main__':
    main()