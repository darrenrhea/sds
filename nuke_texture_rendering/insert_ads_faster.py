from get_camera_posed_actual_annotations import (
     get_camera_posed_actual_annotations
)
from get_repo_ids_to_use_for_fake_data import (
     get_repo_ids_to_use_for_fake_data
)
from get_augmentation_for_texture import (
     get_augmentation_for_texture
)
from AdTextureSource import (
     AdTextureSource
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
from get_approved_annotations_from_these_repos import (
     get_approved_annotations_from_these_repos
)
from pathlib import Path
import numpy as np
import PIL
import PIL.Image
from CameraParameters import CameraParameters
from open_as_rgba_hwc_np_u8 import (
     open_as_rgba_hwc_np_u8
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from render_ads_on_subregion import (
     render_ads_on_subregion
)
from get_world_coordinate_descriptors_of_ad_placements import (
     get_world_coordinate_descriptors_of_ad_placements
)
from draw_euroleague_landmarks import (
     draw_euroleague_landmarks
)
from feathered_paste_for_images_of_the_same_size import (
     feathered_paste_for_images_of_the_same_size
)
from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
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


def insert_ads_faster():
    segmentation_convention = "led_not_led"
    repo_ids_to_use = get_repo_ids_to_use_for_fake_data(
        floor_id="munich",
        segmentation_convention=segmentation_convention
    )
    approved_annotations = get_camera_posed_actual_annotations(
        repo_ids_to_use=repo_ids_to_use
    )

    anti_aliasing_factor = 1

    

    ad_placement_descriptors = get_world_coordinate_descriptors_of_ad_placements(
        clip_id="munich2024-01-09-1080i-yadif",
        with_floor_as_giant_ad=True,
        overcover_by=0.2
    )

    shared_dir = get_the_large_capacity_shared_directory()
    geometry = get_euroleague_geometry()
    points = geometry["points"]

    do_draw_landmarks_to_prove_camera_is_good = False

    photograph_width_in_pixels = 1920
    photograph_height_in_pixels = 1080

    # get an albumentations transform for the texture:
    albu_transform = get_augmentation_for_texture()

    geometry = dict()
    geometry["points"] = points

    fake_backgrounds_dir = shared_dir / "fake_backgrounds"
    fake_backgrounds_dir.mkdir(exist_ok=True)

    ad_texture_source = AdTextureSource()

    for annotation_index, annotation in enumerate(approved_annotations):
        annotation_id = annotation["annotation_id"]
        original_file_path = annotation["original_file_path"]
        mask_file_path = annotation["mask_file_path"]
        clip_id = annotation["clip_id"]
        frame_index = annotation["frame_index"]
        print(f"{original_file_path.name=}")
        print(f"{annotation_index=} out of {len(approved_annotations)}")

        camera_parameters = get_camera_pose_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index
        )
        assert isinstance(camera_parameters, CameraParameters)

        original_rgb_np_u8 = open_as_rgb_hwc_np_u8(
            image_path=original_file_path,
            scale=anti_aliasing_factor
        )
        
        if do_draw_landmarks_to_prove_camera_is_good:
            draw_euroleague_landmarks(
                original_rgb_np_u8=original_rgb_np_u8,
                camera_pose=camera_parameters
            )

        mask_rgba_np_u8 = open_as_rgba_hwc_np_u8(
            image_path=mask_file_path
        )

        original_rgba_np_u8 = np.zeros(
            (original_rgb_np_u8.shape[0], original_rgb_np_u8.shape[1], 4),
            dtype=np.uint8
        )
        original_rgba_np_u8[:, :, :3] = original_rgb_np_u8
        original_rgba_np_u8[:, :, 3] = 255
        
        ijs = np.argwhere(mask_rgba_np_u8[:, :, 3] < 255)

        texture_rgba_np_f32 = ad_texture_source.get_a_random_ad_texture_rgba_np_f32(
            albu_transform=albu_transform
        )

        for ad_placement_descriptor in ad_placement_descriptors:
            ad_placement_descriptor.texture_rgba_np_f32 = texture_rgba_np_f32
        
        rgba_values_at_those_ijs = render_ads_on_subregion(
            ad_placement_descriptors=ad_placement_descriptors,
            ijs=ijs,
            photograph_width_in_pixels=photograph_width_in_pixels,  # needed to convert ijs to normalized [-1,1] x [9/16, 9/16] normalized coordinates
            photograph_height_in_pixels=photograph_height_in_pixels,
            camera_parameters=camera_parameters,
        )
        # place them in 2D:
        ad_placement_accumulator = np.zeros((1080*anti_aliasing_factor, 1920*anti_aliasing_factor, 4), dtype=np.uint8)
        ad_placement_accumulator[ijs[:, 0], ijs[:, 1], :] =  rgba_values_at_those_ijs

        # prii(ad_placement_accumulator)
        
        add_noise_to_render(ad_placement_accumulator)

        composition_rgba_np_uint8 = feathered_paste_for_images_of_the_same_size(
            bottom_layer_color_np_uint8=original_rgb_np_u8,
            top_layer_rgba_np_uint8=ad_placement_accumulator,
        )

        final_pil = PIL.Image.fromarray(composition_rgba_np_uint8)
        antialiased_pil = final_pil.resize(
            (1920, 1080),
            resample=PIL.Image.Resampling.BILINEAR
        )
        bottom_layer_color_np_uint8 = np.array(antialiased_pil)[:, :, :3]
        
        redrawn_foreground = feathered_paste_for_images_of_the_same_size(
            bottom_layer_color_np_uint8=bottom_layer_color_np_uint8,
            top_layer_rgba_np_uint8=mask_rgba_np_u8,
        )

        # prii(original_file_path, caption="this is real:")
        # prii(redrawn_foreground, caption="this is fake:")

        
        # choose where to save the fake annotation:
        rid = np.random.randint(0, 1_000_000_000_000_000)
        fake_annotation_id = f"{annotation_id}_fake{rid:015d}"
        fake_original_out_path = fake_backgrounds_dir / f"{fake_annotation_id}_original.png"
        fake_rgba_out_path = fake_backgrounds_dir / f"{fake_annotation_id}_nonfloor.png"

        # write the fake original for the background:
        write_rgb_np_u8_to_png(
            rgb_hwc_np_u8=redrawn_foreground[:,:, :3],
            out_abs_file_path=fake_original_out_path
        )

        # write the fake mask for the background.  The mask part is unchanged:
        write_rgb_and_alpha_to_png(
            rgb_hwc_np_u8=redrawn_foreground,
            alpha_hw_np_u8=mask_rgba_np_u8[:, :, 3],
            out_abs_file_path=fake_rgba_out_path
        )


if __name__ == "__main__":
    insert_ads_faster()