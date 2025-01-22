from prii import (
     prii
)
from grirosffmi_get_ram_in_ram_out_segmenter_from_final_model_id import (
     grirosffmi_get_ram_in_ram_out_segmenter_from_final_model_id
)
import argparse
import textwrap
import os
from faiac_flatten_and_infer_and_compose import (
     faiac_flatten_and_infer_and_compose
)


def sfabiftf_show_flattened_ad_board_inferences_for_these_frames_cli_tool():

    argp = argparse.ArgumentParser(
        description="show some flattened-ad-board inferences for some frames",
        usage=textwrap.dedent(
            """
            export m=brewcubflattenedvip4
            export c=brewcub
            siftf_show_inferences_for_these_frames 143341
            """
        )
    )

    argp.add_argument(
        "frame_indices",
        nargs="+",
        type=int,
        help="a list of frame indices to infer. The final_model_id is the environment variable m and the clip_id is the environment variable c",
    )
    
    args = argp.parse_args()
    
    frame_indices = args.frame_indices

    try:  
        final_model_id = os.environ["m"]
    except KeyError:
        raise ValueError("please set the environment variable m to the final_model_id like:\n\nexport m=brewcubflattenedvip4")
    
    try:
        clip_id = os.environ["c"]
    except KeyError:
        raise ValueError("please set the environment variable c to the clip_id like:\n\nexport c=brewcub")

    assert clip_id in ["brewcub"], "This only works for baseball because it is homography_based"

    ram_in_ram_out_segmenter = grirosffmi_get_ram_in_ram_out_segmenter_from_final_model_id(
        final_model_id=final_model_id
    )

    for frame_index in frame_indices:
        composition_rgb_hwc_np_u8 = faiac_flatten_and_infer_and_compose(
            ram_in_ram_out_segmenter=ram_in_ram_out_segmenter,
            clip_id=clip_id,
            frame_index=frame_index,
            board_ids=["left", "right"],
            board_id_to_rip_height={"left": 256, "right": 256},
            board_id_rip_width={"left": 1024, "right": 1024},
            verbose=True,
        )
        prii(composition_rgb_hwc_np_u8)
