from convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device import (
     convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device
)
from get_datapoint_path_tuples_for_testing import (
     get_datapoint_path_tuples_for_testing
)
from pathlib import Path
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
from get_cuda_devices import get_cuda_devices

from Segmenter import (
     Segmenter
)
import pprint as pp
from torchvision import transforms

def test_Segmenter():

    which_resolution = "1856x256"
    # which_resolution = "1920x1088"

    if which_resolution == "1920x1088":
        fn_checkpoint = "/shared/checkpoints/u3fasternets-floor-1865frames-1920x1088-wednesday-nba2024finalgame5_epoch000697.pt"
        model_architecture_id = "u3fasternets"
        original_height = 1080
        original_width = 1920
        pad_height = 8
        pad_width = 0
        patch_width = 1920
        patch_height = 1088
        inference_width = 1920
        inference_height = 1088
        patch_stride_width = 1920
        patch_stride_height = 1088
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
        list_of_input_and_output_file_paths = \
        get_list_of_input_and_output_file_paths_from_jsonable(
            jsonable=jsonable,
            out_dir=out_dir_path,
            model_id_suffix=model_id_suffix
        )
    
        pp.pprint(list_of_input_and_output_file_paths)
        number_of_frames = len(list_of_input_and_output_file_paths)
        

    elif which_resolution == "1856x256":
        pad_height = 0
        pad_width = 0
        patch_width = 1856
        patch_height = 256
        inference_width = 1856
        inference_height = 256
        patch_stride_width = 1856
        patch_stride_height = 256
        model_id_suffix = ""
        original_height = 256
        original_width = 1856
        model_architecture_id = "effl"
        weights_file_path = Path(
            "/shared/checkpoints/effl-flatled-323frames-1856x256-jesus_epoch000310.pt"
        )
        assert weights_file_path.is_file()
        
        
        list_of_input_file_paths = [
            x for x, y, z in get_datapoint_path_tuples_for_testing()
        ]
    else:
        raise ValueError(f"which_resolution {which_resolution} not recognized")

    devices = get_cuda_devices()
    num_devices = len(devices)
    assert num_devices > 0, "no cuda devices found"
    device = devices[0]

    out_dir_path = get_a_temp_dir_path()

    segmenter = Segmenter(
        device=device,
        fn_checkpoint=weights_file_path,
        model_architecture_id=model_architecture_id,
        original_height=original_height,
        original_width=original_width,
        inference_height=inference_height,
        inference_width=inference_width,
        pad_height=pad_height,
        pad_width=pad_width,
        patch_height=patch_height,
        patch_width=patch_width,
        patch_stride_height=patch_stride_height,
        patch_stride_width=patch_stride_width,
    )

    for input_file_path in list_of_input_file_paths:
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
    test_Segmenter()