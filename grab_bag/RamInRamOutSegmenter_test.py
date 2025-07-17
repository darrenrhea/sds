import cv2
from load_frame import (
     load_frame
)
from RamInRamOutSegmenter import (
     RamInRamOutSegmenter
)
from pathlib import Path
from prii import (
     prii
)
from get_a_temp_dir_path import (
     get_a_temp_dir_path
)
from get_list_of_input_and_output_file_paths_from_jsonable import (
     get_list_of_input_and_output_file_paths_from_jsonable
)
from get_cuda_devices import get_cuda_devices
import pprint as pp


def test_nonparallel_segment():

    # which_resolution = "1856x256"
    which_resolution = "1920x1088"
    # which_resolution = "1024x256"

    if which_resolution == "1920x1088":
        fn_checkpoint = "/shared/checkpoints/u3fasternets-floor-1271frames-1920x1088-wednesday-nba2024finalgame4_epoch000098.pt"
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
        fn_checkpoint = "/shared/checkpoints/effl-flatled-10000frames-1856x256-plus0_epoch000280.pt"
        jsonable  = {
            "input_dir": "/shared/flattened_training_data",
            "clip_id": "bos-mia-2024-04-21-mxf",
            "original_suffix": "_original.jpg",
            "frame_ranges": [
                370500,
                805500,
            ]
        }
    elif which_resolution == "1024x256":
        pad_height = 0
        pad_width = 0
        patch_width = 512
        patch_height = 256
        inference_width = 1024
        inference_height = 256
        patch_stride_width = 256
        patch_stride_height = 256
        model_id_suffix = ""
        original_height = 256
        original_width = 1024
        model_architecture_id = "effl"
        fn_checkpoint = "/shared/checkpoints/effl-flatled-1256frames-512x256-somefake_epoch000013.pt"
        jsonable  = {
            "input_dir": "/shared/clips/brewcub/flat/left/1024x256/frames",
            "clip_id": "brewcub_left",
            "original_suffix": "_original.png",
            "frame_ranges": [
                250687,
            ]
        }
    else:
        raise ValueError(f"which_resolution {which_resolution} not recognized")

    devices = get_cuda_devices()
    num_devices = len(devices)
    assert num_devices > 0, "no cuda devices found"
    
    out_dir_path = get_a_temp_dir_path()

    list_of_input_and_output_file_paths = \
    get_list_of_input_and_output_file_paths_from_jsonable(
        jsonable=jsonable,
        out_dir=out_dir_path,
        model_id_suffix=model_id_suffix
    )

    for input_file_path, output_file_path in list_of_input_and_output_file_paths:
        assert isinstance(input_file_path, Path)
        assert input_file_path.is_file()
        assert isinstance(output_file_path, Path)

    pp.pprint(list_of_input_and_output_file_paths)
    number_of_frames = len(list_of_input_and_output_file_paths)

  
    devices = get_cuda_devices()
    device = devices[0]

    ram_in_ram_out_segmenter = RamInRamOutSegmenter(
        device=device,
        fn_checkpoint=fn_checkpoint,
        model_architecture_id=model_architecture_id,
        inference_height=inference_height,
        inference_width=inference_width,
        pad_height=pad_height,
        pad_width=pad_width,
        patch_height=patch_height,
        patch_width=patch_width,
        patch_stride_height=patch_stride_height,
        patch_stride_width=patch_stride_width,
    )

    for input_file_path, output_file_path in list_of_input_and_output_file_paths:
        frame_bgr = load_frame(
            frame_path=input_file_path,
            inference_width=inference_width,
            inference_height=inference_height
        )

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mask = ram_in_ram_out_segmenter.infer(
            frame_rgb=frame_rgb
        )

        prii(frame_rgb)
        prii(mask)


if __name__ == "__main__":
    test_nonparallel_segment()