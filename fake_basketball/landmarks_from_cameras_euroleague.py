from get_original_frame_from_clip_id_and_frame_index import (
     get_original_frame_from_clip_id_and_frame_index
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
from pathlib import Path
from nuke_lens_distortion import nuke_world_to_pixel_coordinates
import numpy as np
from Drawable2DImage import Drawable2DImage

clip_id = "munich2024-01-25-1080i-yadif"

frame_indices = {
    "munich2024-01-25-1080i-yadif": [
        146500,
        147000,
    ],
    "munich2024-01-09-1080i-yadif": [
        38500,
    ],
}[clip_id]


shared_dir = get_the_large_capacity_shared_directory()
geometry = get_euroleague_geometry()
points = geometry["points"]
landmark_names = [key for key in points.keys()]


draw_it = True


photograph_width_in_pixels = 1920
photograph_height_in_pixels = 1080


geometry = dict()
geometry["points"] = points


output_dir = Path(
    "temp"
).resolve()

output_dir.mkdir(exist_ok=True)


for frame_index in frame_indices:
    camera_parameters = get_camera_pose_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index
    )

    pixel_points = dict()

    for landmark_name in landmark_names:
        p_giwc = geometry["points"][landmark_name]
    

        x_pixel, y_pixel, is_visible = nuke_world_to_pixel_coordinates(
            p_giwc=np.array(p_giwc),
            camera_parameters=camera_parameters,
            photograph_width_in_pixels=photograph_width_in_pixels,
            photograph_height_in_pixels=photograph_height_in_pixels
        )
        
        if is_visible:
            print(f"    {landmark_name} visible at {x_pixel}, {y_pixel}")
            pixel_points[landmark_name] = dict(
                i=int(y_pixel),
                j=int(x_pixel),
            )
        else:
            pass
            print(f"    {landmark_name} is not visible.")
    
    jsonable = dict(
        pixel_points=pixel_points
    )
    

    if draw_it:        
        rgb_np_uint8 = get_original_frame_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index
        )

        rgba_np_u8 = np.zeros(
            (rgb_np_uint8.shape[0], rgb_np_uint8.shape[1], 4),
            dtype=np.uint8
        )
        rgba_np_u8[:, :, :3] = rgb_np_uint8
        rgba_np_u8[:, :, 3] = 255

        drawable_image = Drawable2DImage(
            rgba_np_uint8=rgba_np_u8,
            expand_by_factor=2
        )

        for landmark_name, dct in pixel_points.items():
            drawable_image.draw_plus_at_2d_point(
                x_pixel=dct["j"],
                y_pixel=dct["i"],
                rgb=(0, 255, 0),
                size=10,
                text=landmark_name
            )
        
        output_image_file_path = Path(f"temp/{clip_id}_{frame_index}.png")
        drawable_image.save(output_image_file_path=output_image_file_path)
        prii(output_image_file_path)
        print(f"pri {output_image_file_path}")

    
    
