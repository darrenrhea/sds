from segmenter_from_final_model_id import (
     segmenter_from_final_model_id
)
import cv2
from load_frame import (
     load_frame
)
from prii import (
     prii
)
from get_a_temp_dir_path import (
     get_a_temp_dir_path
)
from get_list_of_input_and_output_file_paths_from_jsonable import (
     get_list_of_input_and_output_file_paths_from_jsonable
)
import torch


import pprint as pp

def infer_time_augmentation():

    which_resolution = "1856x256"
    which_resolution = "1920x1088"

    if which_resolution == "1920x1088":
        final_model_id = "gamefive697"

     
        inference_width = 1920
        inference_height = 1080
      
        model_id_suffix = ""
        jsonable = {
            "input_dir": "/media/drhea/muchspace/clips/bos-mia-2024-04-21-mxf/frames",
            "clip_id": "bos-mia-2024-04-21-mxf",
            "original_suffix": "_original.jpg",
            "frame_ranges": [
                440694,
                440695,
            ]
        }
       

    elif which_resolution == "1856x256":
        final_model_id = "gamefive697"
        inference_width = 1856
        inference_height = 256
        model_id_suffix = ""
        
        jsonable  = {
            "input_dir": "/shared/flattened_training_data",
            "clip_id": "bos-mia-2024-04-21-mxf",
            "original_suffix": "_original.jpg",
            "frame_ranges": [
                370500,
                805500,
            ]
        }
    else:
        raise ValueError(f"which_resolution {which_resolution} not recognized")

    # check for cuda devices
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    out_dir_path = get_a_temp_dir_path()

    list_of_input_and_output_file_paths = \
    get_list_of_input_and_output_file_paths_from_jsonable(
        jsonable=jsonable,
        out_dir=out_dir_path,
        model_id_suffix=model_id_suffix
    )
  
    pp.pprint(list_of_input_and_output_file_paths)

    segmenter = segmenter_from_final_model_id(
        final_model_id=final_model_id,
    )

    for input_file_path, output_file_path in list_of_input_and_output_file_paths:
        frame_bgr = load_frame(
            frame_path=input_file_path,
            inference_width=inference_width,
            inference_height=inference_height
        )

        

        rgb_hwc_np_u8 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mask_hw_np_u8 = segmenter.infer_rgb_hwc_np_u8_to_hw_np_u8(
            rgb_hwc_np_u8
        )
        
        prii(rgb_hwc_np_u8)

        prii(mask_hw_np_u8)


if __name__ == "__main__":
    infer_time_augmentation()