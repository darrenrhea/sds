from validate_one_camera_pose import (
     validate_one_camera_pose
)
from get_click_on_image_by_two_stage_zoom import (
     get_click_on_image_by_two_stage_zoom
)
from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
from get_original_frame_from_clip_id_and_frame_index import (
     get_original_frame_from_clip_id_and_frame_index
)
import numpy as np

from g3dpaap_get_3d_position_assuming_a_plane import (
    g3dpaap_get_3d_position_assuming_a_plane
)

from prii import (
     prii
)


def g3dpaap_get_3d_position_assuming_a_plane_cli_tool():
    """Get 3D position assuming a plane (CLI tool)."""
    import argparse
    argparser = argparse.ArgumentParser(
        description="Get 3D position assuming a plane.",
        usage="clip_id frame_index plane_name"
    )
    argparser.add_argument("clip_id", type=str, help="Clip ID.")
    argparser.add_argument("frame_index", type=int, help="Frame index.")
    argparser.add_argument("plane_name", type=str, help="Plane name.")
    args = argparser.parse_args()
    clip_id = args.clip_id
    frame_index = args.frame_index
    plane_name = args.plane_name

    original_rgb_hwc_np_u8 = get_original_frame_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index
    )


    validate_camera_pose_image = validate_one_camera_pose(
        clip_id=clip_id,
        frame_index=frame_index,
    )

    prii(
        validate_camera_pose_image,
        caption="the landmarks better line up or the camera pose is wrong:"
    )


    camera_pose = get_camera_pose_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )

    if plane_name == "floor":
        implicit_plane = np.array(
            [0, 0, 1, 0],
            dtype=np.float64
        )
    elif plane_name == "led":
        u = np.array([1, 0, 0])
        v = np.array(
           [
                0.0,
                0.2557832826252587,
                0.9667341477001042
            ]
        )
        
        point_in_plane = np.array(
            [
                0.12828140218734774,
                28.98753407892227,
                1.552085395399928
            ],
            dtype=np.float64
        )

        normal_to_the_plane = np.cross(u, v)
        implicit_plane = np.zeros(4, dtype=np.float64)
        implicit_plane[:3] = normal_to_the_plane
        implicit_plane[3] = -np.dot(normal_to_the_plane, point_in_plane)
    else:
        raise ValueError(f"Unknown plane_name: {plane_name}")

    click = get_click_on_image_by_two_stage_zoom(
        max_display_width=800,
        max_display_height=800,
        rgb_hwc_np_u8=original_rgb_hwc_np_u8,
    )

    if click is None:
        print("No click.")
        return
    x_pixel, y_pixel = click

    # x_pixel = 461.75
    # y_pixel = 303.875

    ans = g3dpaap_get_3d_position_assuming_a_plane(
        xy_pixel_point=[x_pixel, y_pixel],
        original_rgb_hwc_np_u8=original_rgb_hwc_np_u8,
        camera_pose=camera_pose,
        implicit_plane=implicit_plane,
    )
    print(f"{ans=}")
    