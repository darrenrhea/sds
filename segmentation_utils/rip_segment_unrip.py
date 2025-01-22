"""
Given a frame and solved camera:
- Rip left lane, reflect to canonical RHS birds-eye view
- Apply segmentation
- Project segmentation mask back to TV view

- Rip right lane to RHS birds-eye view
- Apply segmentation
- Project segmentation mask back to TV view

- Combine TV view segmentation masks from LHS and RHS
"""

import sys
from pathlib import Path
import os
from PIL import Image, ImageOps
import numpy as np
import better_json as bj
from CameraParameters import CameraParameters
import nuke_texture_rendering

def segment(path_to_image, path_to_output):
    img = Image.open(path_to_image).convert("RGBA")
    # SEGMENT IT
    #img = Image.open( Path("~/awecom/data/clips/gsw1/fake_segmentation_data/0_players.png").expanduser() ) # for testing
    img.save(path_to_output)
    return 

if __name__ == "__main__":

    path_to_frame = Path(sys.argv[1]).expanduser()
    path_to_camera = Path(sys.argv[2]).expanduser()

    pad = 2
    x_bnds = dict(
        left = [-47-pad, -28.17+pad],
        right = [28.17-pad, 47+pad]
    )
    y_min = -7.83 - pad
    y_max = 7.83 + pad

    tmp_dir = Path("./tmp")
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / "ripped_lane.png"
    output_width = 720

    # rip left lane and then right lane
    frame = Image.open(path_to_frame) # only needed for width and height info
    composition_rgba_np_uint8 = np.zeros((frame.height, frame.width, 4)).astype(np.uint8)
    for side in ["left", "right"]:
        x_min, x_max = x_bnds[side]
        #exec_string = "~/r/floor-ripper/build/bin/rip_floor.exe %s %s %s %s %s %s %s %s" % (path_to_camera, x_min, x_max, y_min, y_max, output_width, masked_frame_path, output_path)
        exec_string = "~/r/floor-ripper/rip_floor.exe %s %s %s %s %s %s %s %s" % (path_to_camera, x_min, x_max, y_min, y_max, output_width, path_to_frame, tmp_path)
        os.system(exec_string)
        if side=="left": # reflect LHS into RHS canonical form
            im = Image.open(tmp_path)
            im_mirror = ImageOps.mirror(im)
            im_mirror.save(tmp_path)

        # segment the canonicalized image
        seg_path = tmp_dir / "ripped_lane_seg.png" 
        segment(
            path_to_image = tmp_path, 
            path_to_output = seg_path
        )

        # reproject into the frame
        im = Image.open(seg_path)
        if side=="left": # reflect LHS into RHS canonical form
            im = ImageOps.mirror(im)
            im.save(tmp_path)

        camera_json = bj.load(path_to_camera)
        mask = np.array(im)[:,:,-1] # mask is alpha channel
        camera_parameters = CameraParameters.from_dict(camera_json)
        texture_rgba_np_float32 = np.repeat(mask[:,:,None], 4, axis=2).astype(np.float32)
        composition_rgba_np_uint8 += nuke_texture_rendering.partial_render(
            photograph_width_in_pixels=frame.width,
            photograph_height_in_pixels=frame.height,
            original_photo_rgba_np_uint8=None,
            camera_parameters=camera_parameters,
            texture_rgba_np_float32=texture_rgba_np_float32,
            texture_x_min_wc=x_min,
            texture_x_max_wc=x_max,
            texture_y_min_wc=y_min,
            texture_y_max_wc=y_max,
            i_min=0,
            i_max=frame.height,
            j_min=0,
            j_max=frame.width,
        )
        im_mask = Image.fromarray(composition_rgba_np_uint8) 
        frame.putalpha(im_mask.convert("L"))
        frame.save("./tmp/output.png")