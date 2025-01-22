from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
from encode_indicator_vector_as_list_of_intervals import (
     encode_indicator_vector_as_list_of_intervals
)
from eabbfasf_extract_ad_board_backgrounds_from_a_single_frame import (
     eabbfasf_extract_ad_board_backgrounds_from_a_single_frame
)
from get_original_frame_from_clip_id_and_frame_index import (
     get_original_frame_from_clip_id_and_frame_index
)
from prii import (
     prii
)



def edub_extract_discovered_unoccluded_background(
    clip_id: str,
    frame_index: int,
    rip_height: int,
    rip_width: int,
    min_width: int,
):
    """
    Given that overerasing of the occluders is available
    for the frame in question, extract the discovered unoccluded background(s),
    if any exist.
    """
    assert isinstance(clip_id, str)
    assert isinstance(frame_index, int)
    assert isinstance(rip_height, int)
    assert isinstance(rip_width, int)
    assert isinstance(min_width, int)

    camera_pose = get_camera_pose_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )
    if camera_pose.f == 0:
        print(f"Skipping frame {clip_id}_{frame_index} because it is a non-annotated frame.\n\n\n\n")
        return None

    print(f"{clip_id}_{frame_index:06d}:")
    original_rgb_hwc_np_u8 = get_original_frame_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )

    prii(original_rgb_hwc_np_u8)

    (
        flattened_rgb,
        flattened_overcover_mask,
        visibility_mask
    ) = eabbfasf_extract_ad_board_backgrounds_from_a_single_frame(
        clip_id=clip_id,
        frame_index=frame_index,
        board_id="board0",
        rip_height=rip_height,
        rip_width=rip_width,
    )

    prii(flattened_rgb)
    prii(flattened_overcover_mask)
    prii(visibility_mask)

    # minimize over each column to go one dimensional
    is_column_good = visibility_mask.min(axis=0) >= 255
    a_b_pairs = encode_indicator_vector_as_list_of_intervals(is_column_good)
    # print(a_b_pairs)
    
    wide_enough = [
        (a, b)
        for a, b in a_b_pairs if b - a >= min_width
    ]
    print(f"Found {len(wide_enough)} wide-enough backgrounds:")

    for i, (a, b) in enumerate(wide_enough):
        print(f"Background {i}, {b-a} pixels wide by {rip_height} pixels tall:")
        prii(flattened_rgb[:, a:b, :])
    

   

