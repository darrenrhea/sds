from write_hw_np_u16_to_16_bit_grayscale_png import (
     write_hw_np_u16_to_16_bit_grayscale_png
)
from sha256_of_file import (
     sha256_of_file
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from prii_rgb_and_alpha import (
     prii_rgb_and_alpha
)
from prii import (
     prii
)
from get_local_file_paths_for_annotations import (
     get_local_file_paths_for_annotations
)
from prii_hw_np_nonlinear_u16 import (
     prii_hw_np_nonlinear_u16
)
import shutil
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


def cmottd_compare_multiple_outputs_to_training_data():
    print_in_iterm2 = False
    out_dir = Path("/shared/predicted").expanduser()
    out_dir.mkdir(exist_ok=True, parents=True)
    # There is a relatively small file that describes all the metadata about ALL annotations:
    video_frame_annotations_metadata_sha256 = (
        "4bffcd3e6d1e6cdc0055fdf5004b498e1f07282ddeebe2d524a59d28726208d2"
    )

    desired_labels = set(["depth_map", "floor_not_floor", "original"])

    local_file_pathed_annotations = get_local_file_paths_for_annotations(
        video_frame_annotations_metadata_sha256=video_frame_annotations_metadata_sha256,
        desired_labels=desired_labels,
        max_num_annotations=None,
        print_in_iterm2=False,
    )


    for a in local_file_pathed_annotations:
        # print("")
        # pprint.pprint(local_file_pathed_annotations)
        assert a["local_file_paths"]["original"].is_file()
        assert a["local_file_paths"]["floor_not_floor"].is_file()
        assert a["local_file_paths"]["depth_map"].is_file()
        

    which_resolution = "1920x1088"
   
    if which_resolution == "1920x1088":
        fn_checkpoint = "/shared/checkpoints/u3fasternets-floor-844frames-1920x1088-floornotfloordepthmap_epoch000800.pt"
        num_outputs = 2
        model_architecture_id = "u3fasternets"
        pad_height = 8
        pad_width = 0
        patch_width = 1920
        patch_height = 1088
        inference_width = 1920
        inference_height = 1088
        patch_stride_width = 1920
        patch_stride_height = 1088
        
     
    else:
        raise ValueError(f"which_resolution {which_resolution} not recognized")

    list_of_input_and_should_bes = []
    for annotation in local_file_pathed_annotations:
        local_file_paths = annotation["local_file_paths"]
        input_file_path = local_file_paths["original"]
        
        should_bes = [
            local_file_paths["floor_not_floor"],
            local_file_paths["depth_map"],
        ]

        list_of_input_and_should_bes.append((input_file_path, should_bes))


    devices = get_cuda_devices()
    num_devices = len(devices)
    assert num_devices > 0, "no cuda devices found"
    

    for input_file_path, should_bes in list_of_input_and_should_bes:
        assert isinstance(input_file_path, Path)
        assert input_file_path.is_file()
        assert isinstance(should_bes, list)
        for should_be_file_path in should_bes:
            assert isinstance(should_be_file_path, Path)
            assert should_be_file_path.is_file()
  
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

    for input_file_path, should_bes in list_of_input_and_should_bes:
        assert isinstance(input_file_path, Path)
        assert input_file_path.is_file()
        assert isinstance(should_bes, list)

        for should_be_file_path in should_bes:
            assert isinstance(should_be_file_path, Path)
            assert should_be_file_path.is_file()

        local_file_paths = dict(
            original=input_file_path,
        )

        original_sha256 = sha256_of_file(input_file_path)
        original_suffix = input_file_path.suffix
        
        shutil.copy(
            src=input_file_path,
            dst=out_dir / f"{original_sha256}-original{original_suffix}",
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
        i = 0  # floor_not_floor mask
       
        # prii_hw_np_nonlinear_u16(outputs_chw_np_u16[i, :, :])
        mask_hw_np_u16 = outputs_chw_np_u16[0, :1080, :]

        write_hw_np_u16_to_16_bit_grayscale_png(
            hw_np_u16=mask_hw_np_u16,
            out_abs_file_path=out_dir / f"{original_sha256}-floor_not_floor.png",
            verbose=True,
            display_image_in_iterm2=False,
        )
        if print_in_iterm2:
            alpha_hw_np_u8 = (mask_hw_np_u16 // 256).astype(np.uint8)
            rgb_hwc_np_u8 = open_as_rgb_hwc_np_u8(input_file_path)
            print("mask created by model:")
            prii_rgb_and_alpha(
                rgb_hwc_np_u8=rgb_hwc_np_u8,
                alpha_hw_np_u8=alpha_hw_np_u8,
            )
            should_be_file_path = should_bes[i]
            print("should_be:")
            prii(should_be_file_path)

        i = 1
        should_be_file_path = should_bes[1]
        shutil.copy(
            src=should_be_file_path,
            dst=out_dir / f"{original_sha256}-depth_map_should_be_this.png",
        )
       
        depth_map_hw_np_u16 = outputs_chw_np_u16[1, :1080, :]
        

        write_hw_np_u16_to_16_bit_grayscale_png(
            hw_np_u16=depth_map_hw_np_u16,
            out_abs_file_path=out_dir / f"{original_sha256}-depth_map.png",
            verbose=True,
            display_image_in_iterm2=False,
        )

        if print_in_iterm2:
            print("depth_map created by model:")
            prii_hw_np_nonlinear_u16(depth_map_hw_np_u16)
            print("should_be:")
            prii(should_be_file_path)

            mask = outputs_chw_np_u16[0, :, :]
            depth_map = outputs_chw_np_u16[1, :, :]
            discrete_mask = mask > 32767
            combined = depth_map * discrete_mask.astype(np.uint16)
            prii(input_file_path)
            print("combined mask and depthmap:")
            prii_hw_np_nonlinear_u16(combined)
       

        


if __name__ == "__main__":
    cmottd_compare_multiple_outputs_to_training_data()