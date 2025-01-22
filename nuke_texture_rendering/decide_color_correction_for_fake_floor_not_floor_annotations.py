import textwrap
from prii_linear_f32 import (
     prii_linear_f32
)
from save_color_correction_as_json import (
     save_color_correction_as_json
)
from color_correct_rgb_via_polynomial_coefficients_linear_f32_in_and_out import (
     color_correct_rgb_via_polynomial_coefficients_linear_f32_in_and_out
)
from get_color_correction_polynomial_coefficients_from_from_to_mapping_array_f64 import (
     get_color_correction_polynomial_coefficients_from_from_to_mapping_array_f64
)
from convert_u8_to_linear_f32 import (
     convert_u8_to_linear_f32
)
from get_from_to_mapping_array import (
     get_from_to_mapping_array
)
from open_alpha_channel_image_as_a_single_channel_grayscale_image import (
     open_alpha_channel_image_as_a_single_channel_grayscale_image
)
import sys
from get_descriptor_of_placement_for_nba_floor_texture import (
     get_descriptor_of_placement_for_nba_floor_texture
)
from pathlib import Path
from get_mask_hw_np_u8_from_clip_id_and_frame_index_and_convention_and_final_model_id import (
     get_mask_hw_np_u8_from_clip_id_and_frame_index_and_convention_and_final_model_id
)
from get_original_frame_from_clip_id_and_frame_index import (
     get_original_frame_from_clip_id_and_frame_index
)
from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
from get_rgba_hwc_np_f32_from_texture_id import (
     get_rgba_hwc_np_f32_from_texture_id
)
from insert_quads_into_camera_posed_image_behind_mask import (
     insert_quads_into_camera_posed_image_behind_mask
)
from prii import (
     prii
)
import numpy as np


def make_fake_floor_not_floor_annotations_by_inserting_a_new_floor_underneath():
    """
    Insert a new floor for which models do not work underneath the players etc. in a floor-not-floor annotated frame.
    """

    # It is actually pretty hard to find games that have both floor_not_floor annotations and tracking info.

    clip_id = "hou-sas-2024-10-17-sdi"
    segmentation_convention = "floor_not_floor"
    frame_index = 247900  # this frame has to have a floor_not_floor annotation
    final_model_id = "human"  # means not a model at all but human annotation.
    
    camera_pose = get_camera_pose_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )

    if camera_pose.f == 0.0:
        print("camera_pose.f is 0.0, so we are not going to be able to do anything.")
        sys.exit(1)

    texture_id = "24-25_HOU_CORE"

    uncorrected_texture_rgba_np_linear_f32 = (
        get_rgba_hwc_np_f32_from_texture_id(
            texture_id=texture_id,
            use_linear_light=True
        )
    )

    prii_linear_f32(
        x=uncorrected_texture_rgba_np_linear_f32,
        caption="this is the uncorrected floor texture",
    )

    floor_placement_descriptor = get_descriptor_of_placement_for_nba_floor_texture()


    original_rgb_np_u8 = get_original_frame_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )
    # move to linear light:
    original_rgb_np_linear_f32 = convert_u8_to_linear_f32(
        original_rgb_np_u8
    )

    out_dir=Path("~/ff").expanduser()
    out_dir.mkdir(exist_ok=True)

    prii(
        x=original_rgb_np_u8,
        caption="this is the original frame:",
        out=out_dir / "a.png",
    )

    mask_hw_np_u8 = get_mask_hw_np_u8_from_clip_id_and_frame_index_and_convention_and_final_model_id(
        clip_id=clip_id,
        frame_index=frame_index,
        segmentation_convention=segmentation_convention,
        final_model_id=final_model_id,
    )
    assert mask_hw_np_u8 is not None
    assert isinstance(mask_hw_np_u8, np.ndarray)
    prii(mask_hw_np_u8)
    
    textured_ad_placement_descriptors = []
    floor_placement_descriptor.texture_rgba_np_f32 = uncorrected_texture_rgba_np_linear_f32
    textured_ad_placement_descriptors.append(floor_placement_descriptor)
    assert len(textured_ad_placement_descriptors) == 1

    inserted_without_color_correction_linear_f32 = insert_quads_into_camera_posed_image_behind_mask(
        anti_aliasing_factor=2,
        use_linear_light=True, # this should be true, because of all the averaging processes going on.
        original_rgb_np_linear_f32=original_rgb_np_linear_f32,
        camera_pose=camera_pose,
        mask_hw_np_u8=mask_hw_np_u8,
        textured_ad_placement_descriptors=textured_ad_placement_descriptors,
    )
    assert inserted_without_color_correction_linear_f32.dtype == np.float32


    prii_linear_f32(
        x=inserted_without_color_correction_linear_f32,
        caption="this is the final product of inserting quads into the camera-posed image behind the mask:",
        out=out_dir / "b.png",
    )

    prii(
        x=original_rgb_np_u8,
        caption="this is the original frame:",
    )

    color_sample_mask_path = Path(
        "~/hou-sas-2024-10-17-sdi_247900_colorsamplemask.png"
    ).expanduser()

    color_sampling_mask_hw_np_u8 = open_alpha_channel_image_as_a_single_channel_grayscale_image(
        abs_file_path=color_sample_mask_path
    )
    color_sampling_mask_hw_np_u8[...] = 255 - color_sampling_mask_hw_np_u8[...]

    prii(
        x=color_sampling_mask_hw_np_u8,
        caption="this is the color sampling mask where color will be sampled from:",
    )

    from_to_mapping_array_f64 = get_from_to_mapping_array(
        from_rgb_np=inserted_without_color_correction_linear_f32,
        to_rgb_np=original_rgb_np_linear_f32,
        color_sampling_mask_hw_np_u8=color_sampling_mask_hw_np_u8,
    )
    from_white_u8 = [209, 211, 209]
    to_white_u8 = [245, 226, 249]
    
    repeats = from_to_mapping_array_f64.shape[0]

    black_to_black = np.zeros(
        shape=(
            from_to_mapping_array_f64.shape[0],
            2,
            3
        ),
        dtype=np.float64
    )

    white_to_white_u8 = np.repeat(
        axis=0,
        a=np.array(
            [[from_white_u8, to_white_u8]],
            dtype=np.uint8
        ),
        repeats=repeats
    )
    assert white_to_white_u8.shape[1] == 2
    assert white_to_white_u8.shape[2] == 3

    white_to_white = convert_u8_to_linear_f32(
        white_to_white_u8
    )

    from_to_mapping_array_f64 = np.concatenate( 
        (
            from_to_mapping_array_f64,
            black_to_black,
            white_to_white,
        ),
        axis=0
    )

    degree = 1

    coefficients = get_color_correction_polynomial_coefficients_from_from_to_mapping_array_f64(
        degree=degree,
        from_to_mapping_array_f64=from_to_mapping_array_f64,
    )
    

    color_correction_out_path = Path.home() / "color_correction.json"
    
    save_color_correction_as_json(
        degree=degree,
        coefficients=coefficients,
        out_path=color_correction_out_path
    )

    print(f"saved color correction to {color_correction_out_path}")

    color_corrected_texture_rgb_np_linear_f32 = color_correct_rgb_via_polynomial_coefficients_linear_f32_in_and_out(
        degree=degree,
        coefficients=coefficients,
        rgb_hwc_np_linear_f32=uncorrected_texture_rgba_np_linear_f32[:, :, :3],
    )

    color_corrected_texture_rgba_np_linear_f32 = np.zeros(
        shape=(
            color_corrected_texture_rgb_np_linear_f32.shape[0],
            color_corrected_texture_rgb_np_linear_f32.shape[1],
            4
        ),
        dtype=np.float32
    )
    color_corrected_texture_rgba_np_linear_f32[:, :, :3] = color_corrected_texture_rgb_np_linear_f32
    
   
    prii_linear_f32(
        x=color_corrected_texture_rgb_np_linear_f32,
        caption="this is the color corrected floor texture",
        out=out_dir / "color_corrected_texture.png",
    )

    prii(
        x=original_rgb_np_u8,
        caption="this is the original frame:",
    )

    textured_ad_placement_descriptors = []
    floor_placement_descriptor.texture_rgba_np_f32 = color_corrected_texture_rgba_np_linear_f32
    textured_ad_placement_descriptors.append(floor_placement_descriptor)
    assert len(textured_ad_placement_descriptors) == 1

    inserted_with_color_correction_linear_f32 = insert_quads_into_camera_posed_image_behind_mask(
        anti_aliasing_factor=2,
        use_linear_light=True, # this should be true, because of all the averaging processes going on.
        original_rgb_np_linear_f32=original_rgb_np_linear_f32,
        camera_pose=camera_pose,
        mask_hw_np_u8=mask_hw_np_u8,
        textured_ad_placement_descriptors=textured_ad_placement_descriptors,
    )
    assert inserted_with_color_correction_linear_f32.dtype == np.float32


    prii_linear_f32(
        x=inserted_with_color_correction_linear_f32,
        caption="this is the final color corrected result",
        out=out_dir / "b.png",
    )

    prii(
        x=original_rgb_np_u8,
        caption="this is the original frame:",
    )


    s = textwrap.dedent(
        """
        rm -f ~/ff/*
        rsync -rP 'dl:~/ff/' ~/ff/
        flipflop ~/ff
        """
    )
    print("We suggest you run the following commands:")
    print(s)


if __name__ == "__main__":
    make_fake_floor_not_floor_annotations_by_inserting_a_new_floor_underneath()
   