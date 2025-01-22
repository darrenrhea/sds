from add_an_opaque_alpha_channel_to_create_rgba_hwc_np_f32 import (
     add_an_opaque_alpha_channel_to_create_rgba_hwc_np_f32
)
from typing import List
from convert_linear_f32_to_u8 import (
     convert_linear_f32_to_u8
)
from prii_linear_f32 import (
     prii_linear_f32
)
from convert_u8_to_linear_f32 import (
     convert_u8_to_linear_f32
)
from AdPlacementDescriptor import AdPlacementDescriptor
from add_noise_via_jpg_lossyness import (
     add_noise_via_jpg_lossyness
)
from insert_quads_into_camera_posed_image_behind_mask import (
     insert_quads_into_camera_posed_image_behind_mask
)
import numpy as np
from pathlib import Path
from prii import (
     prii
)


def insert_fake_ads_they_sent_to_us(
    original_rgb_hwc_np_u8: np.ndarray,
    mask_hw_np_u8: np.ndarray,
    camera_pose: np.ndarray,
    ad_placement_descriptors: List[AdPlacementDescriptor],
    final_color_ad_rgb_np_linear_f32: np.ndarray,  # this goes in as is
    verbose: bool,
    flip_flop_dir: Path,
) -> np.ndarray:
    """
    summer_league_2024 has much less reflection we think, so we are dropping that for now.
    We will use it for ads that need color correction.
    """

    final_color_ad_rgba_np_linear_f32 = (
        add_an_opaque_alpha_channel_to_create_rgba_hwc_np_f32(
            final_color_ad_rgb_np_linear_f32
        )
    )

    original_rgb_np_linear_f32 = convert_u8_to_linear_f32(original_rgb_hwc_np_u8)

    
    textured_ad_placement_descriptors = []
    for ad_placement_descriptor in ad_placement_descriptors:
        ad_placement_descriptor.texture_rgba_np_f32 = final_color_ad_rgba_np_linear_f32
        textured_ad_placement_descriptors.append(ad_placement_descriptor)
    
    
    ad_inserted_rgb_np_linear_f32 = insert_quads_into_camera_posed_image_behind_mask(
        use_linear_light=True,
        original_rgb_np_linear_f32=original_rgb_np_linear_f32,
        mask_hw_np_u8=mask_hw_np_u8,
        camera_pose=camera_pose,
        textured_ad_placement_descriptors=textured_ad_placement_descriptors,
        anti_aliasing_factor=3,
    )

    if verbose:
        print("This is the result of inserting the ad:")
        prii_linear_f32(ad_inserted_rgb_np_linear_f32[:, :, :3])

    rgba_np_linear_f32 = np.concatenate(
        [
            ad_inserted_rgb_np_linear_f32,
            mask_hw_np_u8[:, :, np.newaxis].astype(np.float32) / 255.0
        ],
        axis=2
    )
    if verbose:
        prii_linear_f32(
            rgba_np_linear_f32[:, :, :3],
            caption="rgba_np_linear_f32.png",
            out=flip_flop_dir / "rgba_np_linear_f32.png"
        )
    temp_rgb_hwc_np_u8 = convert_linear_f32_to_u8(rgba_np_linear_f32[:, :, :3])
    enshitified_rgb_np_u8 = add_noise_via_jpg_lossyness(
        rgb_hwc_np_u8=temp_rgb_hwc_np_u8,
        jpeg_quality=70,
    )
    if verbose:
        prii(
            enshitified_rgb_np_u8,
            caption="enshitified_rgb_np_u8.png",
            out=flip_flop_dir / "enshitified_rgb_np_u8.png"
        )
    
    return enshitified_rgb_np_u8
