import copy
from AdPlacementDescriptor import (
     AdPlacementDescriptor
)
from insert_quads_into_camera_posed_image_behind_mask import (
     insert_quads_into_camera_posed_image_behind_mask
)
import textwrap
from pathlib import Path
import numpy as np
from prii import (
     prii
)
from CameraParameters import (
     CameraParameters
)


def display_alignment_quality(
    original_rgb_np_u8: np.ndarray,
    camera_pose: CameraParameters,
    mask_for_regression_hw_np_u8: np.ndarray,
    ad_placement_descriptor: AdPlacementDescriptor,
    texture_rgba_np_f32: np.ndarray,
    check_alignment_dir: Path,
) -> np.ndarray:
    """
    This visually displays,
    for consumption by a human,
    how well the ad is aligned by the camera pose and quad placement.
    This draws a textured quad or quads, usually an LED board ad, into the camera-posed image,
    but only drawing where the mask is on (say 255 to be safe).
    """

    height = original_rgb_np_u8.shape[0]
    width = original_rgb_np_u8.shape[1]
    
    textured_ad_placement_descriptor = copy.deepcopy(ad_placement_descriptor)
    textured_ad_placement_descriptor.texture_rgba_np_f32 = texture_rgba_np_f32
    
    overwritten_with_its_own_ad = insert_quads_into_camera_posed_image_behind_mask(
        original_rgb_np_u8=original_rgb_np_u8,
        camera_pose=camera_pose,
        mask_hw_np_u8=255-mask_for_regression_hw_np_u8,
        textured_ad_placement_descriptors=[textured_ad_placement_descriptor,],
        anti_aliasing_factor=2,
        use_linear_light=False,
    )
    
    original_out_path = check_alignment_dir / "original.png"

    prii(
        original_rgb_np_u8,
        caption="this is the original video frame:",
        out=original_out_path,
    )

    overwritten_out_path = check_alignment_dir / "overwritten_sans_color_correction.png"

    prii(
        overwritten_with_its_own_ad,
        caption="this is augmented with its own ad without color correction:",
        out=overwritten_out_path,
    )

    print(
        textwrap.dedent(
            """\
            You should check alignment, because without good spacial alignment
            the regression will not work well.
            Do this:
            """
        )
    )
 
    original_for_regression = np.zeros(
        shape=(height, width, 4),
        dtype=np.uint8
    )

    original_for_regression[:, :, :3] = original_rgb_np_u8
    original_for_regression[:, :, 3] = mask_for_regression_hw_np_u8

    prii(
        original_for_regression,
        caption="this is the original, masked down to what is used for regression",
        out=check_alignment_dir / "a.png"
    )

    overwritten_for_regression = np.zeros(
        shape=(height, width, 4),
        dtype=np.uint8
    )
    overwritten_for_regression[:, :, :3] = overwritten_with_its_own_ad
    overwritten_for_regression[:, :, 3] = mask_for_regression_hw_np_u8

    prii(
        overwritten_for_regression,
        caption="this is it augmented by the same ad as it was, masked down to what is used for regression",
        out=check_alignment_dir / "b.png"
    )

    print(
        textwrap.dedent(
            f"""\
            flipflop {check_alignment_dir}
            """
        )
    )

