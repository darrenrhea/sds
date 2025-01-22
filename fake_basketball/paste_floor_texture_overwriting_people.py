from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
from get_camera_posed_background_frames_but_no_masks import (
     get_camera_posed_background_frames_but_no_masks
)
from augment_texture import (
     augment_texture
)
import pprint as pp

from get_world_coordinate_descriptors_of_ad_placement_for_london_floor_texture import (
     get_world_coordinate_descriptors_of_ad_placement_for_london_floor_texture
)
from get_world_coordinate_descriptors_of_ad_placement_for_munich_floor_texture import (
    get_world_coordinate_descriptors_of_ad_placement_for_munich_floor_texture  
)
from get_augmentation_for_texture import (
     get_augmentation_for_texture
)
from write_rgb_and_alpha_to_png import (
     write_rgb_and_alpha_to_png
)
from write_rgb_np_u8_to_png import (
     write_rgb_np_u8_to_png
)
from add_noise_to_render import (
     add_noise_to_render
)
from pathlib import Path
import numpy as np
import PIL
import PIL.Image
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from render_ads_on_subregion import (
     render_ads_on_subregion
)
from draw_euroleague_landmarks import (
     draw_euroleague_landmarks
)
from feathered_paste_for_images_of_the_same_size import (
     feathered_paste_for_images_of_the_same_size
)
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
from get_euroleague_geometry import (
     get_euroleague_geometry
)
from prii import (
     prii
)


def paste_floor_texture_overwriting_people(
    context_id: str
):
    """
    You do need camera parameters for this, but you dont need a segmentation.
    Best we can do right now is paste the London floor texture OVER the Munich
    frames, for which we have camera parameters.  We can't paste the
    the London floor texture UNDER the the people of NBA until we find camera poses
    for the NBA frames.
    """
    valid_context_ids = ["munich", "london"]

    assert context_id in valid_context_ids, f"ERROR: {context_id=} not in {valid_context_ids=}"


    if context_id == "munich":
        
        floor_texture_file_path = Path(
            "~/r/floortextures/EB_23-24_MUN_floortexture_filled.png"
        ).expanduser()
    elif context_id == "london":
        floor_texture_file_path = Path(
            "~/r/floortextures/BRT_23-24_LDN_floortexture_filled.png"
        ).expanduser()
    else:
        # floor_texture_sha256 = "394608b9239a74b874003419c239f9fd8caf3753d568e4cef7c0b1059ecdab41"
        floor_texture_sha256 = "879a8b1cf59689de76eb94b07a641d97e94162aba5c974cc631254b8a93c7917"
        
        floor_texture_file_path = get_file_path_of_sha256(
            sha256=floor_texture_sha256,
            check=True,
        )
    

    shared_dir = get_the_large_capacity_shared_directory()

    fake_backgrounds_dir = shared_dir / f"fake_{context_id}_people_free_backgrounds"
    fake_backgrounds_dir.mkdir(exist_ok=True)

    approved_annotations = get_camera_posed_background_frames_but_no_masks(
        clip_id="MUN_ASVEL_CALIB_VID",
        original_suffix=".jpg",
        first_frame_index=0,
        last_frame_index=1000,
        step=100
    )

    anti_aliasing_factor = 2

    if context_id == "munich":
        ad_placement_descriptors = \
        get_world_coordinate_descriptors_of_ad_placement_for_munich_floor_texture()
    elif context_id == "london":
        ad_placement_descriptors = \
        get_world_coordinate_descriptors_of_ad_placement_for_london_floor_texture()

    geometry = get_euroleague_geometry()
    points = geometry["points"]

    # Only do this if you worry the camera poses are wrong, it is only for debugging camera pose problems:
    do_draw_landmarks_to_prove_camera_is_good = False

    photograph_width_in_pixels = 1920
    photograph_height_in_pixels = 1080

    # get an albumentations transform for the texture:
    albu_transform = get_augmentation_for_texture()

    geometry = dict()
    geometry["points"] = points

    output_dir = Path(
        "temp"
    ).resolve()

    output_dir.mkdir(exist_ok=True)

    # there is only one "ad texture", the floor texture for London:
    unaugmented_texture_rgb_np_u8 = open_as_rgb_hwc_np_u8(
        image_path=floor_texture_file_path,
    )

    print_out_the_texture = True
    if print_out_the_texture:
        prii(unaugmented_texture_rgb_np_u8, caption="The floor texture")
    
    approved_annotations = approved_annotations[:2]
    for annotation_index, annotation in enumerate(approved_annotations):
        # if annotation_index < 5:
        #     continue
    
        annotation_id = annotation["annotation_id"]
        original_file_path = annotation["original_file_path"]
        camera_parameters = annotation["camera_pose"]
        print(f"{original_file_path.name=}")
        print(f"{annotation_index=}")


        original_rgb_np_u8 = open_as_rgb_hwc_np_u8(
            image_path=original_file_path,
            scale=anti_aliasing_factor
        )
        
        if do_draw_landmarks_to_prove_camera_is_good:
            draw_euroleague_landmarks(
                original_rgb_np_u8=original_rgb_np_u8,
                camera_pose=camera_parameters
            )

        original_rgba_np_u8 = np.zeros(
            (original_rgb_np_u8.shape[0], original_rgb_np_u8.shape[1], 4),
            dtype=np.uint8
        )
        original_rgba_np_u8[:, :, :3] = original_rgb_np_u8
        original_rgba_np_u8[:, :, 3] = 255
        
        # ijs are every ij in the image:
        ijs = np.argwhere(original_rgba_np_u8[:, :, 3] <= 255)

        pp.pprint(ad_placement_descriptors)

        for ad_placement_descriptor in ad_placement_descriptors:
            texture_rgb_np_u8 = augment_texture(
                rgb_np_u8=unaugmented_texture_rgb_np_u8,
                transform=albu_transform
            )

            texture_rgba_np_f32 = np.zeros(
                shape=(
                    texture_rgb_np_u8.shape[0],
                    texture_rgb_np_u8.shape[1],
                    4
                ),
                dtype=np.float32
            )
            texture_rgba_np_f32[:, :, :3] = texture_rgb_np_u8
            texture_rgba_np_f32[:, :, 3] = 255.0

            ad_placement_descriptor.texture_rgba_np_f32 = texture_rgba_np_f32
        
        rgba_values_at_those_ijs = render_ads_on_subregion(
            ad_placement_descriptors=ad_placement_descriptors,
            ijs=ijs,
            photograph_width_in_pixels=photograph_width_in_pixels * anti_aliasing_factor,  # needed to convert ijs to normalized [-1,1] x [9/16, 9/16] normalized coordinates
            photograph_height_in_pixels=photograph_height_in_pixels * anti_aliasing_factor,
            camera_parameters=camera_parameters,
        )
        # place them in 2D:
        ad_placement_accumulator = np.zeros(
            shape=(
                photograph_height_in_pixels * anti_aliasing_factor,
                photograph_width_in_pixels * anti_aliasing_factor,
                4
            ),
            dtype=np.uint8
        )

        ad_placement_accumulator[ijs[:, 0], ijs[:, 1], :] = rgba_values_at_those_ijs
        prii(ad_placement_accumulator)
        
        add_noise_to_render(ad_placement_accumulator)

        composition_rgb_np_uint8 = feathered_paste_for_images_of_the_same_size(
            bottom_layer_color_np_uint8=original_rgb_np_u8,
            top_layer_rgba_np_uint8=ad_placement_accumulator,
        )

        rgb_pil = PIL.Image.fromarray(composition_rgb_np_uint8)
        antialiased_rgb_pil = rgb_pil.resize(
            (photograph_width_in_pixels, photograph_height_in_pixels),
            resample=PIL.Image.Resampling.BILINEAR
        )
        # the final fake original:
        fake_original_rgb = np.array(antialiased_rgb_pil)

        alpha_pil = PIL.Image.fromarray(255 - ad_placement_accumulator[:, :, 3])

        antialiased_alpha_pil = alpha_pil.resize(
            (photograph_width_in_pixels, photograph_height_in_pixels),
            resample=PIL.Image.Resampling.BILINEAR
        )
        # the final fake mask:
        alpha_hw_np_u8 = np.array(antialiased_alpha_pil)

        prii(fake_original_rgb, caption="this is the resulting original")
        prii(alpha_hw_np_u8, caption="this is the resulting alpha")
      
        
        # choose where to save the fake annotation:
        rid = np.random.randint(0, 1_000_000_000_000_000)
        fake_annotation_id = f"{annotation_id}_fake{rid:015d}"
        fake_original_out_path = fake_backgrounds_dir / f"{fake_annotation_id}_original.png"
        fake_rgba_out_path = fake_backgrounds_dir / f"{fake_annotation_id}_nonfloor.png"

        # write the fake original for the background:
        write_rgb_np_u8_to_png(
            rgb_hwc_np_u8=fake_original_rgb,
            out_abs_file_path=fake_original_out_path
        )

        # write the fake mask for the background.  The mask is trivial, we basically are trying to make a people-free-background:
        write_rgb_and_alpha_to_png(
            rgb_hwc_np_u8=fake_original_rgb,
            alpha_hw_np_u8=alpha_hw_np_u8,
            out_abs_file_path=fake_rgba_out_path
        )


if __name__ == "__main__":
    paste_floor_texture_overwriting_people(
        context_id="munich"
    )