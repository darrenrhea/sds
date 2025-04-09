import pprint
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
from CameraParameters import (
     CameraParameters
)
import sys
from clip_id_to_league import (
     clip_id_to_league
)

from get_enough_landmarks_to_validate_camera_pose import (
     get_enough_landmarks_to_validate_camera_pose
)
from draw_named_3d_points import (
     draw_named_3d_points
)
from get_video_frame_path_from_clip_id_and_frame_index import (
     get_video_frame_path_from_clip_id_and_frame_index
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
import better_json as bj

from prii import prii
from scipy.spatial.transform import Rotation as R
import numpy as np

def angles_to_rotation_matrix(angles):
    """
    yaw, pitch, roll angles in radians
    """
    CY = np.cos(angles[0])
    SY = np.sin(angles[0])
    CP = np.cos(angles[1])
    SP = np.sin(angles[1])
    CR = np.cos(angles[2])
    SR = np.sin(angles[2])

    M = np.array(
        [
            [CP * CY, CP * SY, SP],
            [SR * SP * CY - CR * SY, SR * SP * SY + CR * CY, - SR * CP],
            [- (CR * SP * CY + SR * SY),  CY * SR - CR * SP * SY, CR * CP],
        ],
        dtype=np.float64
    )
    return M



def rod_from_angles(angles_in_degrees):
    # Euler angles in radians
    angles = np.radians(angles_in_degrees)
    angle_z, angle_y, angle_x = angles
    altR = angles_to_rotation_matrix(angles)
    rotation = R.from_matrix(altR)
    print("altR:\n", altR)

    # Create a rotation object (specify axes order, e.g., 'xyz')
    # rotation = R.from_euler('zyx', [angle_z, angle_y, angle_x])

    # Convert to rotation matrix
    rotation_matrix = rotation.as_matrix()

    print("Rotation Matrix:\n", rotation_matrix)
    # Convert to Rodrigues vector
    rod = rotation.as_rotvec()
    print("Rodrigues Vector:\n", rod)
    return [rod[0], rod[1], rod[2]]


def convert_new_parametrization():
    league = "nba"
    clip_id = "rabat"
    frame_index = [
        1070,
        20690
    ][1]
    # frame_index = int(np.random.randint(0, 90000))
    print(f"{frame_index=}")
    
    track_sha256 = "312159b2488c13aa264d34113a06993ad9ed211d653c2380a225d892fefea252"
    
    jsonlines_path = get_file_path_of_sha256(
        sha256=track_sha256
    )

    camera_poses = bj.load_jsonlines(
        jsonlines_path=jsonlines_path
    )
    new_json = camera_poses[frame_index]
    pprint.pprint(new_json)
    angles_in_degrees = new_json["angles"]
    if angles_in_degrees[0] == 0:
        print("No camera pose found")
        sys.exit(0)
    rod = rod_from_angles(angles_in_degrees)
    loc_in_meters = new_json["loc"]
    f = new_json["f"]
    k1 = new_json["k1"]
    k2 = new_json["k2"]
    #k1=0
    #k2=0 

    loc = [
        loc_in_meters[0] * 3.28084,
        loc_in_meters[1] * 3.28084,
        loc_in_meters[2] * 3.28084,
    ]

    camera_pose = CameraParameters(
        rod=rod,
        loc=loc,
        f=f,
        k1=k1,
        k2=k2,
    )

    original_file_path = get_video_frame_path_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index
    )

    original_rgb_np_u8 = open_as_rgb_hwc_np_u8(
        image_path=original_file_path
    )


    

    landmark_name_to_xyz = get_enough_landmarks_to_validate_camera_pose(
        league=league
    )
    print(camera_pose)
    drawn_on = draw_named_3d_points(
        original_rgb_np_u8=original_rgb_np_u8,
        camera_pose=camera_pose,
        landmark_name_to_xyz=landmark_name_to_xyz
    )
    prii(drawn_on)

if __name__ == "__main__":
    convert_new_parametrization()