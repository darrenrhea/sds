import better_json as bj
from pathlib import Path
import numpy as np
from CameraParameters import CameraParameters
from tripod_homography import (
    convert_pixel_units_homography_to_normalized_units_homography,
    infer_camera_b_focal_length_and_world_to_camera_from_homography
)
from rodrigues_utils import rodrigues_from_SO3

homographies_dir = Path("~/awecom/data/clips/swinney1/homographies").expanduser()

first_frame = 4700

source_camera_parameters_json_path = Path(
    f"~/awecom/data/clips/swinney1/tracking_attempts/chaz_locked/swinney1_{first_frame:06d}_camera_parameters.json"
).expanduser()

camera_parameters_jsonable = bj.load(source_camera_parameters_json_path)
cp_a = CameraParameters.from_dict(camera_parameters_jsonable)


for second_frame in range(4701, 4800):
    homography_path = homographies_dir / f"{first_frame:06d}_into_{second_frame:06d}.json"
    jsonable = bj.load(homography_path)
    H_in_pixel_units = np.array(
         jsonable["homography_in_pixel_units"]
    )

    H_in_normalized_units = convert_pixel_units_homography_to_normalized_units_homography(H_in_pixel_units)
    
    predicted_f_b, predicted_world_to_camera_b = infer_camera_b_focal_length_and_world_to_camera_from_homography(
        cp_a,
        H_in_normalized_units
    )
    predicted_rod_b = rodrigues_from_SO3(predicted_world_to_camera_b)
    
    cp_b = CameraParameters(
        rod=predicted_rod_b,
        loc=cp_a.loc,
        f=predicted_f_b,
        ppi=0,
        ppj=0,
        k1=0,
        k2=0,
        k3=0,
        p1=0,
        p2=0
    )

    output_camera_parameters_json_path = Path(
        f"~/awecom/data/clips/swinney1/tracking_attempts/homography/swinney1_{second_frame:06d}_camera_parameters.json"
    ).expanduser()

    bj.dump(
        fp=output_camera_parameters_json_path,
        obj=cp_b.to_dict()
    )
    
