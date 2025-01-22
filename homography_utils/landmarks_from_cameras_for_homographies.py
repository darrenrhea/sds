"""
There are fairly trustable cameras for swinney1
ranging from frame_index 4600 to frame_index 7300 saved in
lam:~/awecom/data/clips/swinney1/tracking_attempts/chaz_locked/
We take these supposedly perfect camera fits for these video frames
and a list of known-location-landmarks
and make some JSONs, one per video frame,
that say where the landmarks you want to train HRNet for
are visible if they are visible, usually some are off screen.

{
    "pixel_points": {
        "tl_ral_tip_m": {
            "i": 732.0,
            "j": 554.0
        },
        "tl_cnr_i": {
            "i": 597.0,
            "j": 740.0
        },
        "bl_ll_at_el_i": {
            "i": 815.0,
            "j": 163.0
        },
        "tl_ll_at_el_i": {
            "i": 711.0,
            "j": 439.0
        },
        "bl_fl_at_ll_ii": {
            "i": 857.0,
            "j": 994.0
"""

from pathlib import Path
import better_json as bj
from camera_solve_contexts import get_geometry
from CameraParameters import CameraParameters
from nuke_lens_distortion import nuke_world_to_pixel_coordinates
import numpy as np
from Drawable2DImage import Drawable2DImage
import numpy as np
import PIL

draw_it = True

geometry = get_geometry("ncaa_kansas")

# landmark_names = bj.load("hrnet_point_list.json")
landmark_names = [
    "tl_cnr_i",
    "tr_cnr_i",

    "tl_ll_at_el_i",
    "bl_ll_at_el_i",
    "bl_fl_at_ll_ii",
    "tl_fl_at_ll_ii",

    "tr_ll_at_el_i",
    "br_ll_at_el_i",
    "br_fl_at_ll_ii",
    "tr_fl_at_ll_ii",
    "l_elbow",
    "r_elbow",
    "cpt_3",
    "cpt_2",
]

clip_ids = [
    "swinney1",
]

photograph_width_in_pixels = 1920
photograph_height_in_pixels = 1080

video_clip_id_to_frame_indices_for_that_video_clip = dict(
    swinney1=[
        4613,
        4700,
        4800,
        4900
    ]
)

for clip_id in clip_ids:
    output_dir = Path(
        f"~/awecom/data/clips/{clip_id}/landmark_locations_for_homographies"
    ).expanduser()
    output_dir.mkdir(exist_ok=True)
    frame_indices = video_clip_id_to_frame_indices_for_that_video_clip[clip_id]
    for frame_index in frame_indices:
        pixel_points = dict()

        camera_parameters_path = Path(
            f"~/awecom/data/clips/{clip_id}/tracking_attempts/chaz_locked/{clip_id}_{frame_index:06d}_camera_parameters.json"
        ).expanduser()

        original_image_path = Path(
            f"~/awecom/data/clips/{clip_id}/frames/{clip_id}_{frame_index:06d}.jpg"
        ).expanduser()

        output_json_path = output_dir / f"{clip_id}_{frame_index:06d}_landmark_locations_for_homographies.json"

        camera_parameters = CameraParameters.from_dict(
            bj.load(camera_parameters_path)
        )

        for landmark_name in landmark_names:
            p_giwc = geometry["points"][landmark_name]
        

            x_pixel, y_pixel, is_visible = nuke_world_to_pixel_coordinates(
                p_giwc=p_giwc,
                camera_parameters=camera_parameters,
                photograph_width_in_pixels=photograph_width_in_pixels,
                photograph_height_in_pixels=photograph_height_in_pixels
            )
            
            if is_visible:
                # print(f"    {landmark_name} visible at {x_pixel}, {y_pixel}")
                pixel_points[landmark_name] = dict(
                    i=int(y_pixel),
                    j=int(x_pixel),
                )
            else:
                pass
                # print(f"    {landmark_name} is not visible.")
        jsonable = dict(
            pixel_points=pixel_points
        )
        
        bj.dump(fp=output_json_path, obj=jsonable)
        print(output_json_path)
        if draw_it:
            original_image_pil = PIL.Image.open(str(original_image_path)).convert("RGBA")
            rgba_np_uint8 = np.array(original_image_pil)
            drawable_image = Drawable2DImage(
                rgba_np_uint8=rgba_np_uint8,
                expand_by_factor=1
            )

            for landmark_name, dct in pixel_points.items():
                drawable_image.draw_plus_at_2d_point(
                    x_pixel=dct["j"], y_pixel=dct["i"], rgb=(255, 128, 128), size=1, text=landmark_name
                )
            output_image_file_path = Path(f"temp/{clip_id}_{frame_index}.png")
            drawable_image.save(output_image_file_path=output_image_file_path)
            print(f"open {output_image_file_path}")

        
      