from typing import List
from get_cuda_devices import (
     get_cuda_devices
)
from RamInRamOutSegmenter import (
     RamInRamOutSegmenter
)
from get_original_frame_from_clip_id_and_frame_index import (
     get_original_frame_from_clip_id_and_frame_index
)
import sys
from get_video_frame_path_from_clip_id_and_frame_index import (
     get_video_frame_path_from_clip_id_and_frame_index
)
import shutil

import numpy as np
from woatw_white_out_above_the_wall import (
     woatw_white_out_above_the_wall
)
from write_rgb_and_alpha_to_png import (
     write_rgb_and_alpha_to_png
)
from open_alpha_channel_image_as_a_single_channel_grayscale_image import (
     open_alpha_channel_image_as_a_single_channel_grayscale_image
)
from print_red import (
     print_red
)
from pathlib import Path
from gpksafasf_get_primary_keyed_segmentation_annotations_from_a_single_folder import (
     gpksafasf_get_primary_keyed_segmentation_annotations_from_a_single_folder
)
import pprint

def setup_model():

    fn_checkpoint = "/shared/checkpoints/u3fasternets-vip-251frames-1920x1088-chicago4k_1080p_epoch001536.pt"
    model_architecture_id = "u3fasternets"
    pad_height = 8
    pad_width = 0
    patch_width = 1920
    patch_height = 1088
    inference_width = 1920
    inference_height = 1088
    patch_stride_width = 1920
    patch_stride_height = 1088
    
    

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
    
    return ram_in_ram_out_segmenter

def mwatwtdfmc_make_white_above_the_wall_training_data_for_mathieu_caller(
    clip_id,
    frame_indices: List[int],
):
    ram_in_ram_out_segmenter = setup_model()

    out_dir = Path(
        "/shared/brewcub_whiteabovethewall/unassigned3"
    )
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # grab the vip_segmentation_convention annotations from this folder:
    dataset_folder = Path(
        "~/r/brewcub_vip/.unknown"
    ).expanduser()
    
    assert dataset_folder.exists(), f"{dataset_folder=} does not exist"

    annotations = gpksafasf_get_primary_keyed_segmentation_annotations_from_a_single_folder(
        dataset_folder=dataset_folder,
        diminish_to_this_many=None,
    )

    print(f"{len(annotations)=}")

    for frame_index in frame_indices:
        # pprint.pprint(dct)
        # clip_id = dct["clip_id"]
        # frame_index = dct["frame_index"]
        # mask_path = dct["mask_path"]
        # dont_use_this_original_path = dct["original_path"]
        # mask_out_abs_file_path = out_dir / mask_path.name
        mask_out_abs_file_path = out_dir / f"{clip_id}_{frame_index:06d}_nonfloor.png"
        
        # # extracting at 4K then downsampling to 1080p might not be the same as extracting at 1080p:

        better_original_path = get_video_frame_path_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index
        )

        original_rgb_hwc_np_u8 = get_original_frame_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index,
        )

        # # TODO: masks could also be calculated from the vip model instead of loaded from hard drive:
        # mask_hw_np_u8 = open_alpha_channel_image_as_a_single_channel_grayscale_image(
        #     abs_file_path=mask_path
        # )

        mask_hw_np_u8_padded = ram_in_ram_out_segmenter.infer(
            frame_rgb=original_rgb_hwc_np_u8
        )
        
        mask_hw_np_u8 = mask_hw_np_u8_padded[:1080, :1920]

        assert mask_hw_np_u8.shape == (1080, 1920), f"{mask_hw_np_u8.shape=}"

        mutated_mask_hw_np_u8 = woatw_white_out_above_the_wall(
            clip_id="brewcub",
            frame_index=frame_index,
            mask_hw_np_u8=mask_hw_np_u8,
        )
        
        if mutated_mask_hw_np_u8 is not None:
            num_pixels_changed = np.sum(mask_hw_np_u8 != mutated_mask_hw_np_u8)
        else:
            num_pixels_changed = 0
        
        print(f"{num_pixels_changed=}")
        if mutated_mask_hw_np_u8 is None or num_pixels_changed < 10000:
            print_red(f"skipping {frame_index}")
        else:
            shutil.copy(
                src=better_original_path,
                dst=out_dir
            )
            
            pri(original_rgb_hwc_np_u8)

            write_rgb_and_alpha_to_png(
                rgb_hwc_np_u8=original_rgb_hwc_np_u8,
                alpha_hw_np_u8=mutated_mask_hw_np_u8,
                out_abs_file_path=mask_out_abs_file_path,
                verbose=True
            )


if __name__ == "__main__":

   
    clip_id = "brewcub"
    frame_indices = [
        91127,
        91177,
        91227,
        91277,
        91327,
        91377,
        91427,
        91477,
        91527,
        91577,
        91627,
        91677,
        91727,
        91777,
        91827,
        91877,
    ]
    frame_indices = [
        x
        for x in range(91900, 300700 + 1, 1000)
    ]
    mwatwtdfmc_make_white_above_the_wall_training_data_for_mathieu_caller(
        clip_id=clip_id,
        frame_indices=frame_indices,
    )
