from prii_hw_np_nonlinear_u16 import (
     prii_hw_np_nonlinear_u16
)
import numpy as np
from blackpad_preprocessor_u16_to_u16 import (
     blackpad_preprocessor_u16_to_u16
)
from make_a_channel_stack_u16_from_local_file_pathed_annotation import (
     make_a_channel_stack_u16_from_local_file_pathed_annotation
)
from RamInRamOutSegmenterForMultipleOutputs import (
     RamInRamOutSegmenterForMultipleOutputs
)
from pathlib import Path
from get_cuda_devices import get_cuda_devices
import pprint as pp


def test_RamInRamOutSegmenterForMultipleOutputs_1():

    which_resolution = "1920x1088"
    # which_resolution = "1856x256"
    # which_resolution = "1024x256"

    if which_resolution == "1920x1088":
        fn_checkpoint = "/shared/checkpoints/u3fasternets-floor-844frames-1920x1088-floornotfloordepthmap_epoch000011.pt"
        num_outputs = 2
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
        desired_clip_id_frame_index_pairs=[
            ('SL_2022_00', 133500),
            ('SL_2022_00', 136100),
            ('CHAvNYK_PGM_city_bal_12-09-2022', 267000),
        ],
        list_of_input_and_multiple_output_file_paths_str = [
            (
                # "/media/drhea/muchspace/sha256/db/be/c7/2f/dbbec72faacecd2cef367a1d9fc078674efdaae5d8ff2ea850c2e7e2cc1fa589.jpg",
                # "/mnt/nas/volume1/videos/segmentation_datasets/21-22_bos_core_floor_approved/BOSvMIA_30-03-2022_C01_CLN_MXF_680900.jpg",
                # "/mnt/nas/volume1/videos/segmentation_datasets/21-22_bos_core_floor_approved/BOSvMIA_30-03-2022_C01_CLN_MXF_204000.jpg",
                "/mnt/nas/volume1/videos/segmentation_datasets/hou-sas-2024-10-17-sdi_floor/hou-sas-2024-10-17-sdi_522200_original.jpg",
                (
                    "~/out_floor_not_floor.png",
                    "~/out_depth_map.png",
                ),
            )
        ]

        list_of_input_and_multiple_output_file_paths = [
            (
                Path(input_file_path).expanduser(),
                [Path(output_file_path).expanduser() for output_file_path in output_file_paths]
            )
            for input_file_path, output_file_paths in list_of_input_and_multiple_output_file_paths_str
        ]
    else:
        raise ValueError(f"which_resolution {which_resolution} not recognized")

    devices = get_cuda_devices()
    num_devices = len(devices)
    assert num_devices > 0, "no cuda devices found"
    

    for input_file_path, output_file_paths in list_of_input_and_multiple_output_file_paths:
        assert isinstance(input_file_path, Path)
        assert input_file_path.is_file()
        assert isinstance(output_file_paths, list)
        for output_file_path in output_file_paths:
            assert isinstance(output_file_path, Path)
            assert output_file_path.parent.is_dir()
        
    pp.pprint(list_of_input_and_multiple_output_file_paths)
    number_of_frames = len(list_of_input_and_multiple_output_file_paths)

  
    devices = get_cuda_devices()
    device = devices[0]

    ram_in_ram_out_segmenter_with_multiple_outputs = RamInRamOutSegmenterForMultipleOutputs(
        num_outputs=num_outputs,
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

    for input_file_path, output_file_paths in list_of_input_and_multiple_output_file_paths:
        local_file_paths = dict(
            original=input_file_path,
        )

        local_file_pathed_annotation = dict(
            local_file_paths=local_file_paths
        )
       
        channel_stack_hwc_np_u16 = make_a_channel_stack_u16_from_local_file_pathed_annotation(
            local_file_pathed_annotation=local_file_pathed_annotation,
            channel_names=["r", "g", "b"],
        )
        params = dict(
            desired_height=1088,
            desired_width=1920,
        )
        padded_channel_stack_hwc_np_u16 = blackpad_preprocessor_u16_to_u16(
            channel_stack=channel_stack_hwc_np_u16,
            params=params,
        )

        channel_stack_hwc_np_f32 = padded_channel_stack_hwc_np_u16.astype(np.float32) / 65535.0

        outputs_chw_np_u16 = ram_in_ram_out_segmenter_with_multiple_outputs.infer(
            channel_stack_hwc_np_f32
        )
        for i in range(num_outputs):
            prii_hw_np_nonlinear_u16(outputs_chw_np_u16[i, :, :])


        


if __name__ == "__main__":
    test_RamInRamOutSegmenterForMultipleOutputs_1()