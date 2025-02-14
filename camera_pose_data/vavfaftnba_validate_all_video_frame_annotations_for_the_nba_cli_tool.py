import argparse
from vovfa_validate_one_video_frame_annotation import (
     vovfa_validate_one_video_frame_annotation
)
from gacpvfa_get_all_camera_posed_video_frame_annotations import (
     gacpvfa_get_all_camera_posed_video_frame_annotations
)
import numpy as np
from prii import (
     prii
)


def vavfaftnba_validate_all_video_frame_annotations_for_the_nba_cli_tool():
    """
    To be well-organized, we need to be able to gather all of the 
    NBA basketball video frames for which we have
    * a known-to-be-correct floor_not_floor segmentation, usually a human annotation
    * a known-to-be-correct the camera pose,
    * which floor / court it is, like den_city_2021
    * which teams are playing
    * and what uniforms they are wearing
    * etc.

    No big buffers are here, just sha256s of the big buffers.
    """
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "--draw_a_certificate_image", "-d",
        action="store_true",
    )
    args = argp.parse_args()
    draw_a_certificate_image = args.draw_a_certificate_image
    
    all_camera_posed_video_frame_annotations = (
        gacpvfa_get_all_camera_posed_video_frame_annotations()
    )
    
    np.random.shuffle(
        all_camera_posed_video_frame_annotations
    )
    
    for annotation in all_camera_posed_video_frame_annotations:
        print(annotation)
        drawn_on = vovfa_validate_one_video_frame_annotation(
            annotation=annotation,
            draw_a_certificate_image=draw_a_certificate_image,
        )

        if draw_a_certificate_image:
            prii(
                drawn_on
            )
   