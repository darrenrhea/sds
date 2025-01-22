from find_color_mapping_from_samples import (
     find_color_mapping_from_samples
)
from quantize_colors_via_kmeans import (
     quantize_colors_via_kmeans
)
from gather_colors_on_indicator import (
     gather_colors_on_indicator
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

which_placement = 1
anti_aliasing_factor = 1

clip_id = "munich2024-01-09-1080i-yadif"

ad_placement_descriptors = get_world_coordinate_descriptors_of_ad_placements(
    clip_id=clip_id,
    overcover_by=0.0,

)

# dotflat made this flat directory of good stuff from the approvals.json5:
approved_dir = Path("~/r/munich1080i_led/.approved").expanduser()
print(f"{approved_dir=}")


approved_annotations = get_approved_annotations_from_these_repos()

shared_dir = get_the_large_capacity_shared_directory()
geometry = get_euroleague_geometry()
points = geometry["points"]
landmark_names = [key for key in points.keys()]


do_draw_landmarks_to_prove_camera_is_good = False


photograph_width_in_pixels = 1920
photograph_height_in_pixels = 1080


geometry = dict()
geometry["points"] = points


output_dir = Path(
    "color"
).resolve()

output_dir.mkdir(exist_ok=True)
approved_annotations = approved_annotations[7:8]
ad_placement_descriptors = ad_placement_descriptors[which_placement:which_placement+1]

for annotation_index, annotation in enumerate(approved_annotations):
    original_file_path = annotation["original_file_path"]
    mask_file_path = annotation["mask_file_path"]
    clip_id = annotation["clip_id"]
    frame_index = annotation["frame_index"]
    print(f"{original_file_path.name=}")
    print(f"{annotation_index=}")

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
    
    
    texture_paths = [
        Path("~/r/munich_led_videos/TURKISH AIRLINES/FINAL_THY_New LED boards BAYERN_1920x1080/1152x144/00000.png").expanduser(),
        # Path("~/r/munich_led_videos/ADIDAS/Bayern Render Comp/1016x144/00000.png").expanduser(),
        # Path("~/r/munich_led_videos/BKT/FC BAYERN MUNICH 1920x1080_new/1024x144/00000.png").expanduser(),
        # Path("~/r/munich_led_videos/BETANO/BETANO_BAYERN_HD_201223/1024x144/00979.png").expanduser(),
    ]

    texture_pils = [
        PIL.Image.open(str(texture_path)).convert("RGBA")
        for texture_path in texture_paths  # texture has transparent bits
    ]
    

    texture_rgba_np_f32s = [
        np.array(texture_pil).astype(np.float32)
        for texture_pil in texture_pils
    ]


    ijs = np.argwhere(mask_rgba_np_u8[:, :, 3] < 255)

    # print(ijs)
    # print(f"{ijs.shape=}")
    # visual = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # visual[ijs[:, 0], ijs[:, 1], :] = 255
    # prii(visual)

    # enrich ad_placement_descriptors with random textures:
    for ad_placement_descriptor in ad_placement_descriptors:
        np.random.randint(0, len(texture_rgba_np_f32s))
        texture_rgba_np_f32 = texture_rgba_np_f32s[np.random.randint(0, len(texture_rgba_np_f32s))]
        ad_placement_descriptor.texture_rgba_np_f32 = texture_rgba_np_f32
    
    rgba_values_at_those_ijs = render_ads_on_subregion(
        ad_placement_descriptors=ad_placement_descriptors,
        ijs=ijs,
        photograph_width_in_pixels=1920,  # needed to convert ijs to normalized [-1,1] x [9/16, 9/16] normalized coordinates
        photograph_height_in_pixels=1080,
        camera_parameters=camera_parameters,
    )
    # place them in 2D:
    ad_placement_accumulator = np.zeros((1080, 1920, 4), dtype=np.uint8)
    ad_placement_accumulator[ijs[:, 0], ijs[:, 1], :] =  rgba_values_at_those_ijs

    prii(ad_placement_accumulator, caption="We would like this fake insertion:")
    indicator = ad_placement_accumulator[:,:,3] > 0
    prii(indicator, caption="mask")

    real_masked = np.zeros((1080, 1920, 4), dtype=np.uint8)
    real_masked[indicator, :] = original_rgba_np_u8[indicator, :]

    prii(real_masked, caption="to look like this real image:")

    find_color_mapping_from_samples(
        original_rgba_np_u8=original_rgba_np_u8,
        real_indicator=indicator,
        fake_rgba_np_u8=ad_placement_accumulator,
        fake_indicator=indicator,
    )


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

    prii(
        original_file_path,
        caption="this is real:",
        out=Path(output_dir / f"{annotation_index:06d}_real.png")
    )
    prii(
        redrawn_foreground,
        caption="this is fake:",
        out=Path(output_dir / f"{annotation_index:06d}_fake.png")
    )
   
    
