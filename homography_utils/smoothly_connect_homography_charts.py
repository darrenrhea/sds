"""
The goal here is to smoothly connect homographies and
write the result out as distortionless camera parameters
"""
from pathlib import Path
import numpy as np
import PIL
import PIL.Image
import sys
import better_json as bj
import pprint as pp
from homography_utils import (
    map_points_through_homography, somehow_get_homography_from_video_frame_to_world_floor,
    get_distortionless_camera_parameters_from_homography_from_world_floor_to_photo_pixel
)
from prepare_ads import prepare_ads
from homography_texture_rendering import homography_ad_insertion

from interpolate_keyframes import interpolate_keyframes

def main():
    clip_id = "swinney1"
    
    # homography_to_keyframes_attempt_id = "david08"
    # homography_tracking_attempt_id = "first"
    # first_frame_index = 3700
    # last_frame_index = 5600

    homography_to_keyframes_attempt_id = "felix"
    homography_tracking_attempt_id = "felix"
    first_frame_index = 10000
    last_frame_index = 11000

    draw_ads = True
    if draw_ads:
        ads = prepare_ads()

    keyframe_indices = [
        # 6200 super far left
        4613, # far left
        4700, # midleft
        # 4760  # middle
        4800, # a bit right
        # 4830 midright
        4900, # far right
        # 5410 # super-right
    ]

    # if x <- -20.44 then mainly 4613
    # -20.44 < x < -5.45 then 4700
    # if -5.45 then 4800
    # if x > 11.3 then 4900

    ransacReprojThreshold = 1.0 # ????
    keyframe_index_to_homography = dict()
    for keyframe_index in keyframe_indices:
        H = somehow_get_homography_from_video_frame_to_world_floor(
            keyframe_index,
            ransacReprojThreshold=ransacReprojThreshold
        )
        assert abs(H[2][2] - 1.0) < 1e-6
        keyframe_index_to_homography[keyframe_index] = H

    saved_homographies_dir = Path(f"~/awecom/data/clips/swinney1/homographies_to_keyframes_attempts/{homography_to_keyframes_attempt_id}").expanduser()
    keyframe_index_to_homography_from_photo_to_world_floor = dict()
    keyframe_index_to_weight = dict()

    for frame_index in range(first_frame_index, last_frame_index + 1):
        best_keyframe_index = None
        best_num_matched = 0
        lst_of_homographies = []
        for keyframe_index in keyframe_indices:
            input_json_path = saved_homographies_dir / f"{frame_index:06d}_into_{keyframe_index:06d}.json"
            jsonable = bj.load(input_json_path)
            success = jsonable["success"]
            if success:
                num_matched = jsonable["num_matched"]
                if num_matched > best_num_matched:
                    best_num_matched = num_matched
                    best_keyframe_index = keyframe_index
                homography_3x3 = jsonable["homography_in_pixel_units"]
                homography_from_photo_to_keyframe = np.array(
                    homography_3x3
                )
                
                homography_from_keyframe_to_world_floor = keyframe_index_to_homography[keyframe_index]
                homography_from_photo_to_world_floor = homography_from_keyframe_to_world_floor @ homography_from_photo_to_keyframe
                homography_from_photo_to_world_floor /= homography_from_photo_to_world_floor[2][2]
                keyframe_index_to_homography_from_photo_to_world_floor[keyframe_index] = homography_from_photo_to_world_floor
                lst_of_homographies.append(
                    dict(
                        keyframe_index=keyframe_index,
                        homography_from_photo_to_world_floor=homography_from_photo_to_world_floor,
                        num_matched=num_matched,
                    )
                )
        lst_of_homographies = sorted(lst_of_homographies, key=lambda x: x["num_matched"], reverse=True)
        best_num_matched = lst_of_homographies[0]["num_matched"]
        sndbest_num_matched = lst_of_homographies[1]["num_matched"]
        print(f"best_num_matched = {best_num_matched}, sndbest_num_matched = {sndbest_num_matched}")


        bestH = lst_of_homographies[0]["homography_from_photo_to_world_floor"]
        best_keyframe_index = lst_of_homographies[0]["keyframe_index"]
        sndH = lst_of_homographies[1]["homography_from_photo_to_world_floor"]
        snd_best_keyframe_index = lst_of_homographies[1]["keyframe_index"]
        x_in_the_world1 = map_points_through_homography(
            np.array([[1920/2, 1080]]),
            bestH
        )[0][0]
        pp.pprint(x_in_the_world1)
        x_in_the_world2 = map_points_through_homography(
            np.array([[1920/2, 1080]]),
            sndH
        )[0][0]
        pp.pprint(x_in_the_world2)
        avg_x = (x_in_the_world1 + x_in_the_world2) / 2
        print(f"avg_x = {avg_x}")
        print(f"for {frame_index} best_keyframe = {best_keyframe_index} because {best_num_matched}\n")

        ans = interpolate_keyframes(avg_x)

        frac1 = ans["frac1"]
        frac2 = ans["frac2"]
        keyframe1 = ans["keyframe1"]
        keyframe2 = ans["keyframe2"]

        for keyframe_index in keyframe_indices:
            keyframe_index_to_weight[keyframe_index] = 0.0
        keyframe_index_to_weight[keyframe1] = frac1
        if frac2 > 0:
            keyframe_index_to_weight[keyframe2] = frac2

        avg_homography = np.zeros(shape=(3, 3))
        
        for keyframe_index in keyframe_indices:
            weight = keyframe_index_to_weight[keyframe_index]
            if weight > 0:
                homography_from_photo_to_world_floor = keyframe_index_to_homography_from_photo_to_world_floor[keyframe_index]
                avg_homography += weight * homography_from_photo_to_world_floor


        jsonable = [
            [float(avg_homography[i,j]) for j in range(3)]
            for i in range(3)
        ]
        # recovered = get_distortionless_camera_parameters_from_homography_from_world_floor_to_photo_pixel(
        #     np.linalg.inv(homography_from_photo_to_world_floor)
        # )
        # pp.pprint(recovered)

        out_path = Path(
            f"~/awecom/data/clips/{clip_id}/photo_pixel_to_world_floor_homography_attempts/{homography_tracking_attempt_id}/{clip_id}_{frame_index:06d}_homography.json"
        ).expanduser()
        
        bj.dump(
            fp=out_path,
            obj=jsonable
        )
        
        print(f"see {out_path}")
            
        if draw_ads:  # this is slow.  Let github.com/darrenrhea/homography_video_maker do it.
            original_image_path = Path(
                f"~/awecom/data/clips/swinney1/frames/swinney1_{frame_index:06d}.jpg"
            ).expanduser()

            image_pil = PIL.Image.open(str(original_image_path)).convert("RGBA")
            w_image = image_pil.width
            h_image = image_pil.height
            assert (w_image, h_image) == (1920, 1080)
            original_photo_rgba_np_uint8 = np.array(image_pil)
            assert (
                original_photo_rgba_np_uint8.shape[2] == 4
            ), "original_photo_rgba_np_uint8 should be RGBA thus 4 channels"
            insertion_rgba_np_uint8 = original_photo_rgba_np_uint8

            # insert the ads:
            for ad in ads:
                insertion_rgba_np_uint8 = homography_ad_insertion(
                    photograph_width_in_pixels=1920,
                    photograph_height_in_pixels=1080,
                    original_photo_rgba_np_uint8=insertion_rgba_np_uint8,
                    homography_from_photo_to_world_floor=avg_homography,
                    texture_rgba_np_float32=ad["ad_rgba_np_float32"],
                    texture_x_min_wc=ad["x_center"] - ad["width"]/2,
                    texture_x_max_wc=ad["x_center"] + ad["width"]/2,
                    texture_y_min_wc=ad["y_center"] - ad["height"]/2,
                    texture_y_max_wc=ad["y_center"] + ad["height"]/2
                )
        
       
            final_pil = PIL.Image.fromarray(insertion_rgba_np_uint8).convert("RGB")
            out_dir = Path("~/awecom/data/clips/swinney1/insertion_attempts/homography").expanduser()
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / f"swinney1_{frame_index:06d}_ad_insertion.jpg"
            final_pil.save(out_path)
            print(f"pri {out_path}")



if __name__ == "__main__":
    main()
