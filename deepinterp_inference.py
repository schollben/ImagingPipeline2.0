import os
import sys
import time
import glob
import pathlib
import numpy as np
#import tensorflow as tf
from datetime import datetime
#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession

sys.path.append('C:\\Users\\greg\\Documents\\Python\\deepinterpolation-0.1.5')

from deepinterpolation.cli.inference import Inference

# Forcing CPU usage
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                
def main():
    
    data_paths = [
    #    "D:\\seagate_data_for_ben\\TSeries-08252022-1332-002\\Registered\\Channel1",
        "D:\\seagate_data_for_ben\\TSeries-08252022-1332-003\\Registered\\Channel1",
    #    "D:\\seagate_data_for_ben\\TSeries-08252022-1332-004\\Registered\\Channel1",
        "D:\\seagate_data_for_ben\\TSeries-08252022-1332-005\\Registered\\Channel1"
    #    "D:\\seagate_data_for_ben\\TSeries-08252022-1332-006\\Registered\\Channel1"
    ]

    start_frames = np.repeat(1, len(data_paths))
    end_frames = np.repeat(-1, len(data_paths))
    #start_frames =  [#1,
    #                1,
    #                1,
    #                1,
    #                1
    #]
                
    #end_frames =    [#-1, 
                    #-1,
                    #-1,
                    #-1,
    #                -1
    #]

    assert len(data_paths) == len(start_frames) == len(end_frames)
    
    generator_param = {}
    inference_param = {}

    generator_param["name"] = "OphysGenerator"
    generator_param["batch_size"] = 5
    generator_param["pre_frame"] = 30
    generator_param["post_frame"] = 30
    generator_param["pre_post_omission"] = 0    
    inference_param["output_padding"] = False
    # default is float32
    #inference_param["output_datatype"] = 'uint16'
    
    time_deltas = []
    for i in range(0,len(data_paths)):
        data_maybe = glob.glob(data_paths[i] + "\\registered.h5")
        generator_param["data_path"] = data_maybe[0]
        generator_param["batch_size"] = 5
        generator_param["start_frame"] = start_frames[i]
        generator_param["end_frame"] = end_frames[i]

        inference_param["name"] = "core_inferrence"
        
        model_path = glob.glob(data_paths[i] + "\\*transfer_model.h5")
        inference_param["model_source"] = {
            "local_path": model_path[0]
        }
        #inference_param["output_file"] = '' + data_paths[i] + '\\inference_results_transfer_trained\\res.h5'
        #inference_param["output_file"] = "D:\\transferTraining_v_fineTuning\\delme\\" + str(i) + "\\res.h5"
        #out_path = "D:\\transferTraining_v_fineTuning\\delme\\" + str(i)
        #out_path = data_paths[i] + "\\inference_results.h5"
        #os.mkdir(out_path)
        inference_param["output_file"] = data_paths[i] + "\\inference_results.h5"
        
        args = {
            "generator_params": generator_param,
            "inference_params": inference_param,
            "output_full_args": True
        }

        t_start = datetime.now()
        inference_obj = Inference(input_data=args, args=[])
        inference_obj.run()
        t_stop = datetime.now()
        time_deltas.append(t_stop - t_start)
        print(str(t_stop - t_start))

if __name__ == "__main__":
    main()
