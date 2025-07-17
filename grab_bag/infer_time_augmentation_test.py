from get_flat_original_path import (
     get_flat_original_path
)
from prii_rgb_and_alpha import (
     prii_rgb_and_alpha
)
from infer_time_augmentation import (
     infer_time_augmentation
)
from segmenter_from_final_model_id import (
     segmenter_from_final_model_id
)
import cv2
from load_frame import (
     load_frame
)
from get_a_temp_dir_path import (
     get_a_temp_dir_path
)
from get_list_of_input_and_output_file_paths_from_jsonable import (
     get_list_of_input_and_output_file_paths_from_jsonable
)
import albumentations as A

import pprint as pp

def test_infer_time_augmentation_1():

    augmentation = A.Compose(
        [            
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p = 1.0),
            A.RGBShift(r_shift_limit=12, g_shift_limit=12, b_shift_limit=12, p=1.0),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.2), p=1.0),
        ]
    )

    which_resolution = "1856x256"
    # which_resolution = "1920x1088"

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
        final_model_id = "effs94"
        inference_width = 1856
        inference_height = 256
        model_id_suffix = ""
        
        clip_id_frame_index_pairs = [
            ["bos-mia-2024-04-21-mxf", 388682],
        ]
    else:
        raise ValueError(f"which_resolution {which_resolution} not recognized")


    out_dir_path = get_a_temp_dir_path()

    list_of_input_and_output_file_paths = []
    for clip_id, frame_index in clip_id_frame_index_pairs:
        input_file_path = get_flat_original_path(
            clip_id=clip_id,
            frame_index=frame_index,
            board_id="board0",
            rip_width=1856,
            rip_height=256,
        )
        output_file_path = out_dir_path / f"{clip_id}_{frame_index}_mask.png"
        list_of_input_and_output_file_paths.append((input_file_path, output_file_path))

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

        mask_hw_np_u8 = infer_time_augmentation(
            rgb_hwc_np_u8=rgb_hwc_np_u8,
            segmenter=segmenter,
            augmentation=augmentation,
            verbose=True,
        )

        print("Average mask:")
        prii_rgb_and_alpha(
            rgb_hwc_np_u8=rgb_hwc_np_u8,
            alpha_hw_np_u8=mask_hw_np_u8
        )
        


if __name__ == "__main__":
    test_infer_time_augmentation_1()