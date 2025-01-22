import numpy as np
import json
from pathlib import Path
import PIL
import PIL.Image
from PIL import ImageFont
from PIL import ImageDraw
from computer_quirks import get_font_file_path


def show_keypoint_matches(
    src_image,
    dst_image,
    src_points,
    dst_points,
    src_out_path,
    dst_out_path,
    match_indicator
):
    """
    To be honest, to show keypoint matches,
    the images should be in color.
    Grayscale is a SIFTism.
    """
    src_image_pil = PIL.Image.fromarray(src_image).convert("RGB")
    dst_image_pil = PIL.Image.fromarray(dst_image).convert("RGB")
    draw1 = ImageDraw.Draw(src_image_pil)
    draw2 = ImageDraw.Draw(dst_image_pil)
    font_path = get_font_file_path()
    font = ImageFont.truetype(str(font_path), 20)
    text_color = (0, 255, 0)
    jsonable = []

    for i, is_this_a_match in enumerate(match_indicator):
        if not is_this_a_match:
            point_color = (255, 0, 0)
            text_color = (255, 0, 0)
        else:
            point_color = (0, 255, 128)
            text_color = (0, 255, 128)
        location_in_img1 = src_points[i, :]
        point = (
            int(location_in_img1[0]),
            int(location_in_img1[1]),
        )
        draw1.text(
            xy = point,
            text=f"{i}",
            font=font,
            fill=text_color
        )
        draw1.line(
            xy=[
                (point[0] - 3, point[1]),
                (point[0] + 3, point[1]),
            ],
            fill=point_color
        )
        draw1.line(
            xy=[
                (point[0], point[1] - 3),
                (point[0], point[1] + 3),
            ],
            fill=point_color
        )
        spotting = dict(
            image_id=1,
            tracker_id=i,
            x=float(location_in_img1[0]),
            y=float(location_in_img1[1]),
            homography_match=bool(is_this_a_match)
        )
        jsonable.append(spotting)
    
    labeled_src_image_output_path = str(src_out_path.resolve())
    src_image_pil.save(labeled_src_image_output_path)
    print(f"See {labeled_src_image_output_path}")

    print(f"of which, only {np.sum(match_indicator)} survived RANSAC")
    for i, is_this_a_match in enumerate(match_indicator):
        if not is_this_a_match:
            point_color = (255, 0, 0)
            text_color = (255, 0, 0)
        else:
            point_color = (0, 255, 128)
            text_color = (0, 255, 128)
        location_in_img2 = dst_points[i, :]
        point = (
            int(location_in_img2[0]),
            int(location_in_img2[1]),
        )
        draw2.text(
            xy=point,
            text=f"{i}",
            font=font,
            fill=text_color
        )
        draw2.line(
            xy=[
                (point[0] - 3, point[1]),
                (point[0] + 3, point[1]),
            ],
            fill=point_color
        )
        draw2.line(
            xy=[
                (point[0], point[1]-3),
                (point[0], point[1]+3),
            ],
            fill=point_color
        )
        spotting = dict(
            image_id=2,
            tracker_id=i,
            x=float(location_in_img2[0]),
            y=float(location_in_img2[1]),
            homography_match=bool(is_this_a_match)
        )
        jsonable.append(spotting)

    labeled_dst_image_output_path = str(dst_out_path.resolve())
    dst_image_pil.save(labeled_dst_image_output_path)
    print(f"See {labeled_dst_image_output_path}")

   
    formatted_json = json.dumps(jsonable, indent=4, separators=(", ", ": "))
    # print(formatted_json)
    with open(f"trackers.json", "w") as fp:
        fp.write(formatted_json)

