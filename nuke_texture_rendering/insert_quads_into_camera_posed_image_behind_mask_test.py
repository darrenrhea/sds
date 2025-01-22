from convert_u8_to_linear_f32 import (
     convert_u8_to_linear_f32
)
from prii_linear_f32 import (
     prii_linear_f32
)
from get_mirror_world_ad_placement_descriptor import (
     get_mirror_world_ad_placement_descriptor
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
from get_world_coordinate_descriptors_of_ad_placements import (
     get_world_coordinate_descriptors_of_ad_placements
)
from prii import (
     prii
)
import numpy as np


def test_insert_quads_into_camera_posed_image_behind_mask_1():
    """
    Test by inserting the mirror-world floor
    """
    clip_id = "bos-mia-2024-04-21-mxf"
    frame_index = 734500
    
    texture_ids = [
        "23-24_BOS_CORE",
    ]

    texture_rgba_np_f32s = [
        get_rgba_hwc_np_f32_from_texture_id(texture_id=texture_id, use_linear_light=False)
        for texture_id in texture_ids
    ]
    texture_rgba_np_f32s[0] = np.flip(texture_rgba_np_f32s[0], axis=0)

    camera_pose = get_camera_pose_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )

    mirror_ad_placement_descriptor = get_mirror_world_ad_placement_descriptor(
        clip_id=clip_id,
        with_floor_as_giant_ad=False,
        overcover_by=0.0
    )

    ad_placement_descriptors = [
        mirror_ad_placement_descriptor
    ]

    
    original_rgb_np_u8 = get_original_frame_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )

    out_dir=Path("~/ff").expanduser()
    out_dir.mkdir(exist_ok=True)

    prii(
        x=original_rgb_np_u8,
        caption="this is the original frame:",
        out=out_dir / "a.png",
    )
    
    segmentation_convention = "led"
    final_model_id = "human"
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
    for ad_placement_descriptor, texture_rgba_np_f32 in zip(ad_placement_descriptors, texture_rgba_np_f32s):
        ad_placement_descriptor.texture_rgba_np_f32 = texture_rgba_np_f32
        textured_ad_placement_descriptors.append(ad_placement_descriptor)

    ans = insert_quads_into_camera_posed_image_behind_mask(
        use_linear_light=False,
        original_rgb_np_u8=original_rgb_np_u8,
        camera_pose=camera_pose,
        mask_hw_np_u8=mask_hw_np_u8,
        textured_ad_placement_descriptors=textured_ad_placement_descriptors,
    )

    assert isinstance(ans, np.ndarray)

    prii(
        x=ans,
        caption="this is the final product of inserting quads into the camera-posed image behind the mask:",
        out=out_dir / "b.png",
    )

    print("ff ~/ff")


def test_insert_quads_into_camera_posed_image_behind_mask_2():
    clip_id = "bos-mia-2024-04-21-mxf"
    frame_index = 734500
    
    texture_ids = [
        "different_here",
    ]

    texture_rgba_np_f32s = [
        get_rgba_hwc_np_f32_from_texture_id(texture_id=texture_id, use_linear_light=False)
        for texture_id in texture_ids
    ]

    camera_pose = get_camera_pose_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )

    ad_placement_descriptors = get_world_coordinate_descriptors_of_ad_placements(
        clip_id=clip_id,
        with_floor_as_giant_ad=False,
        overcover_by=0.0,
    )
    
    original_rgb_np_u8 = get_original_frame_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )

    out_dir=Path("~/ff").expanduser()
    out_dir.mkdir(exist_ok=True)

    prii(
        x=original_rgb_np_u8,
        caption="this is the original frame:",
        out=out_dir / "a.png",
    )
    
    segmentation_convention = "led"
    final_model_id = "human"
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
    for ad_placement_descriptor, texture_rgba_np_f32 in zip(ad_placement_descriptors, texture_rgba_np_f32s):
        ad_placement_descriptor.texture_rgba_np_f32 = texture_rgba_np_f32
        textured_ad_placement_descriptors.append(ad_placement_descriptor)

    ans = insert_quads_into_camera_posed_image_behind_mask(
        use_linear_light=False,
        original_rgb_np_u8=original_rgb_np_u8,
        camera_pose=camera_pose,
        mask_hw_np_u8=mask_hw_np_u8,
        textured_ad_placement_descriptors=textured_ad_placement_descriptors,
    )

    assert isinstance(ans, np.ndarray)

    prii(
        x=ans,
        caption="this is the final produGt of inserting quads into the camera-posed image behind the mask:",
        out=out_dir / "b.png",
    )

    print("ff ~/ff")



def test_insert_quads_into_camera_posed_image_behind_mask_3():
    """
    Test by inserting the mirror-world floor
    """
    clip_id = "bos-mia-2024-04-21-mxf"
    frame_index = 734500
    
    texture_ids = [
        "23-24_BOS_CORE",
    ]

    texture_rgba_np_f32s = [
        get_rgba_hwc_np_f32_from_texture_id(texture_id=texture_id, use_linear_light=True)
        for texture_id in texture_ids
    ]
    texture_rgba_np_f32s[0] = np.flip(texture_rgba_np_f32s[0], axis=0)

    camera_pose = get_camera_pose_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )

    mirror_ad_placement_descriptor = get_mirror_world_ad_placement_descriptor(
        clip_id=clip_id,
        with_floor_as_giant_ad=False,
        overcover_by=0.0
    )

    ad_placement_descriptors = [
        mirror_ad_placement_descriptor
    ]

    
    original_rgb_np_u8 = get_original_frame_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )
    prii(
        x=original_rgb_np_u8,
        caption="this is the original frame:",
        out=Path("~/ff/a.png").expanduser(),
    )

    original_rgb_np_linear_f32 = convert_u8_to_linear_f32(original_rgb_np_u8)
    
    segmentation_convention = "led"
    final_model_id = "human"
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
    for ad_placement_descriptor, texture_rgba_np_f32 in zip(ad_placement_descriptors, texture_rgba_np_f32s):
        ad_placement_descriptor.texture_rgba_np_f32 = texture_rgba_np_f32
        textured_ad_placement_descriptors.append(ad_placement_descriptor)

    ans = insert_quads_into_camera_posed_image_behind_mask(
        use_linear_light=True,
        original_rgb_np_linear_f32=original_rgb_np_linear_f32,
        camera_pose=camera_pose,
        mask_hw_np_u8=mask_hw_np_u8,
        textured_ad_placement_descriptors=textured_ad_placement_descriptors,
        anti_aliasing_factor=2,
    )

    assert isinstance(ans, np.ndarray)

    prii_linear_f32(
        x=ans,
        caption="this is the final product of inserting quads into the camera-posed image behind the mask:",
        out=Path("~/ff/b.png").expanduser(),
    )

    print("ff ~/ff")


def test_insert_quads_into_camera_posed_image_behind_mask_4():
    clip_id = "bos-mia-2024-04-21-mxf"
    frame_index = 734500
    
    texture_ids = [
        "different_here",
    ]

    texture_rgba_np_f32s = [
        get_rgba_hwc_np_f32_from_texture_id(texture_id=texture_id, use_linear_light=True)
        for texture_id in texture_ids
    ]

    prii_linear_f32(texture_rgba_np_f32s[0])

    camera_pose = get_camera_pose_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )

    ad_placement_descriptors = get_world_coordinate_descriptors_of_ad_placements(
        clip_id=clip_id,
        with_floor_as_giant_ad=False,
        overcover_by=0.0,
    )
    
    original_rgb_np_u8 = get_original_frame_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )
    original_rgb_np_linear_f32 = convert_u8_to_linear_f32(original_rgb_np_u8)

    prii(
        x=original_rgb_np_u8,
        caption="this is the original frame:",
        out=Path("~/ff/a.png").expanduser(),
    )
    
    segmentation_convention = "led"
    final_model_id = "human"
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
    for ad_placement_descriptor, texture_rgba_np_f32 in zip(ad_placement_descriptors, texture_rgba_np_f32s):
        ad_placement_descriptor.texture_rgba_np_f32 = texture_rgba_np_f32
        textured_ad_placement_descriptors.append(ad_placement_descriptor)

    ans = insert_quads_into_camera_posed_image_behind_mask(
        use_linear_light=True,
        original_rgb_np_linear_f32=original_rgb_np_linear_f32,
        camera_pose=camera_pose,
        mask_hw_np_u8=mask_hw_np_u8,
        textured_ad_placement_descriptors=textured_ad_placement_descriptors,
        anti_aliasing_factor=2,
    )

    assert isinstance(ans, np.ndarray)

    prii_linear_f32(
        x=ans,
        caption="this is the final product of inserting quads into the camera-posed image behind the mask:",
        out=Path("~/ff/b.png").expanduser(),
    )

    print("ff ~/ff")


if __name__ == "__main__":
    test_insert_quads_into_camera_posed_image_behind_mask_1()
    test_insert_quads_into_camera_posed_image_behind_mask_2()
    test_insert_quads_into_camera_posed_image_behind_mask_3()
    test_insert_quads_into_camera_posed_image_behind_mask_4()
