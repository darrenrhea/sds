from get_from_to_mapping_array import (
     get_from_to_mapping_array
)
from prii import (
     prii
)
from convert_u8_to_linear_f32 import (
     convert_u8_to_linear_f32
)
import copy
from AdPlacementDescriptor import (
     AdPlacementDescriptor
)
from insert_quads_into_camera_posed_image_behind_mask import (
     insert_quads_into_camera_posed_image_behind_mask
)
import numpy as np
from CameraParameters import (
     CameraParameters
)


def render_then_get_from_to_mapping_array(
    use_linear_light: bool,
    original_rgb_np_u8: np.ndarray,
    camera_pose: CameraParameters,
    mask_for_regression_hw_np_u8: np.ndarray,
    ad_placement_descriptor: AdPlacementDescriptor,
    texture_rgba_np_f32: np.ndarray,
) -> np.ndarray:
    """
    What this does:
    This draws a textured quad, usually an LED board ad, into the camera-posed image,
    but only drawing where the mask is on (say 255 to be safe).
    Then, the rgb pixel values where the mask is on
    of both the original image and the augmented image are raveled into numpy arrays
    and returns:
    a 2D numpy array of shape (n, 2, 3) where n is the number of pixels where the mask is on.
    """
    height = original_rgb_np_u8.shape[0]
    width = original_rgb_np_u8.shape[1]
    
    textured_ad_placement_descriptor = copy.deepcopy(ad_placement_descriptor)
    textured_ad_placement_descriptor.texture_rgba_np_f32 = texture_rgba_np_f32
    
    if use_linear_light:
        print("use_linear_light is True djdjle")
        original_rgb_linear_f32 = convert_u8_to_linear_f32(original_rgb_np_u8)
   
        overwritten_with_its_own_ad_linear_f32 = insert_quads_into_camera_posed_image_behind_mask(
            use_linear_light=use_linear_light,
            original_rgb_np_linear_f32=original_rgb_linear_f32,
            camera_pose=camera_pose,
            mask_hw_np_u8=255 - mask_for_regression_hw_np_u8,
            textured_ad_placement_descriptors=[textured_ad_placement_descriptor,],
            anti_aliasing_factor=2,
        )
        assert overwritten_with_its_own_ad_linear_f32.dtype == np.float32

    else:
        overwritten_with_its_own_ad = insert_quads_into_camera_posed_image_behind_mask(
            use_linear_light=use_linear_light,
            original_rgb_np_u8=original_rgb_np_u8,
            camera_pose=camera_pose,
            mask_hw_np_u8=255 - mask_for_regression_hw_np_u8,
            textured_ad_placement_descriptors=[textured_ad_placement_descriptor,],
            anti_aliasing_factor=2,
        )
        assert overwritten_with_its_own_ad.dtype == np.uint8
        prii(overwritten_with_its_own_ad, caption="overwritten_with_its_own_ad")
    
    from_to_mapping_array_f64 = get_from_to_mapping_array(
        from_rgb_np=overwritten_with_its_own_ad_linear_f32,
        to_rgb_np=original_rgb_np_linear_f32,
        mask_for_regression_hw_np_u8=mask_for_regression_hw_np_u8,
    )
       
    ijs = np.argwhere(mask_for_regression_hw_np_u8 > 128)

    inputs = overwritten_with_its_own_ad[ijs[:, 0], ijs[:, 1], :]
    outputs_u8 = original_rgb_np_u8[ijs[:, 0], ijs[:, 1], :]
    
    if use_linear_light:
        outputs = convert_u8_to_linear_f32(outputs_u8)
    else:
        inputs = inputs.astype(np.float32) / 255.0
        outputs = outputs_u8 / 255.0

    # form the regression data:
    from_to_mapping_array_f64 = np.zeros(
        shape=(
            ijs.shape[0],
            2,
            3,
        ),
        dtype=np.float64
    )
    from_to_mapping_array_f64[:, 0, :] = inputs
    from_to_mapping_array_f64[:, 1, :] = outputs
    
    assert from_to_mapping_array_f64.shape[0] == ijs.shape[0]
    assert from_to_mapping_array_f64.shape[1] == 2
    assert from_to_mapping_array_f64.shape[2] == 3
    assert from_to_mapping_array_f64.dtype == np.float64

    return from_to_mapping_array_f64

