import better_json as bj
from pathlib import Path
from CameraParameters import CameraParameters
import numpy as np
import PIL
import PIL.Image
import PIL.ImageDraw
from nuke_lens_distortion import nuke_world_to_pixel_coordinates
import pprint as pp


def get_subrectangles_that_need_masking(
    camera_parameters,
    ads,
    drawable
):
    """
    This returns a list of dicts, where each dict defines a rectangle like:
   
        dict(
            i_min=i_min,
            i_max=i_max,
            j_min=j_min,
            j_max=j_max
        )
    such that the subrectangles described cover all the ads visible.
    """
    photograph_width_in_pixels = 1920
    photograph_height_in_pixels = 1080

    list_of_subrectangles = []  # eventually we will return this list of rectangles

    for ad in ads:
        width = ad["width"]
        height = ad["height"]
        x_center = ad["x_center"]
        y_center = ad["y_center"]
        x_min = x_center - width/2
        x_max = x_center + width/2
        y_min = y_center - height/2
        y_max = y_center + height/2

        corners_3d = [
            np.array([x_max, y_max, 0]),
            np.array([x_max, y_min, 0]),
            np.array([x_min, y_min, 0]),
            np.array([x_min, y_max, 0]),
        ]
        line_segments_3d = [
            (
                corners_3d[k],  # a
                corners_3d[(k+1) % 4]  # b
            )
            for
            k in range(4)
        ]

        
        i_min = 20000
        i_max = -20000
        j_min = 20000
        j_max = -20000
        for line_segment in line_segments_3d:
            a, b = line_segment
            for n in range(20 + 1):
                u = n / 20  # u ranges in the interval [0, 1]
                p_giwc = (1-u) * a + u * b
                j, i, is_visible = nuke_world_to_pixel_coordinates(
                    p_giwc=p_giwc,
                    camera_parameters=camera_parameters,
                    photograph_width_in_pixels=1920,
                    photograph_height_in_pixels=1080
                )
                # print(f"j={j}, i={i},  is_visible={is_visible}")
                if not is_visible or i < 0 or i > 1079 or j < 0 or j > 1919:
                    continue
                
                if drawable is not None:
                    color = (0, 255, 0)
                    drawable.line((j, i, j, i), fill=color, width=1)
                i_min = min(i_min, i)
                i_max = max(i_max, i)
                j_min = min(j_min, j)
                j_max = max(j_max, j)
        
        if i_min == 20000:  # this happens because none of the points of that ad are visible:
            continue

        margin = 3
        i_min = max(i_min - margin, 0)
        i_max = min(i_max + margin, 1080)
        j_min = max(j_min - margin, 0)
        j_max = min(j_max + margin, 1920)
        i_min = int(i_min)
        i_max = int(i_max)
        j_min = int(j_min)
        j_max = int(j_max)


        list_of_subrectangles.append(
            dict(
                i_min=i_min,
                i_max=i_max,
                j_min=j_min,
                j_max=j_max,
            )
        )

        if drawable is not None:
            corners_2d = [
                (i_min, j_min),
                (i_max, j_min),
                (i_max, j_max),
                (i_min, j_max)
            ]

            line_segments_2d = [
                (
                    corners_2d[k],  # a
                    corners_2d[(k+1) % 4]  # b
                )
                for
                k in range(4)
            ]

            for line_segment in line_segments_2d:
                a, b = line_segment
                color = (255, 0, 0)  # red
                drawable.line(
                    (a[1], a[0], b[1], b[0]),
                    fill=color,
                    width=1
                )
    return list_of_subrectangles


if __name__ == "__main__":
    # We need descriptions of where to insert the ads into the basketball floor, which is [-47, 47] x [-25, 25] x {0}
    ads = [
        {
            "x_center": -37.6,
            "y_center": 13.2,
            "width": 8,
            "height": 5
        },
        {
            "x_center": 37.6,
            "y_center": 13.2,
            "width": 8,
            "height": 5
        }
    ]

    frame_index = 4881
    # frame_index = 4881
    # frame_index = 4752  # one partially visible ad
    # frame_index = 4767  # no ads are visible, should return a length zero list
    # frame_index = 10450  # really zoomed out, you get two rectangles


    image_path = Path(
        f"~/awecom/data/clips/swinney1/insertion_attempts/chaz/swinney1_{frame_index:06d}_ad_insertion.jpg"
    ).expanduser()

    image_pil = PIL.Image.open(str(image_path))
    image_np = np.array(image_pil)

    camera_parameters_path = Path(
        f"~/awecom/data/clips/swinney1/tracking_attempts/chaz_locked/swinney1_{frame_index:06d}_camera_parameters.json"
    ).expanduser()

    camera_parameters_json = bj.load(camera_parameters_path)
    camera_parameters = CameraParameters.from_dict(camera_parameters_json)
    print(camera_parameters)
    drawable = PIL.ImageDraw.Draw(image_pil)
    list_of_subrectangles = get_subrectangles_that_need_masking(
        camera_parameters=camera_parameters,
        ads=ads,
        drawable=drawable
    )
    
    pp.pprint(list_of_subrectangles)

    image_pil.save("temp.bmp")

