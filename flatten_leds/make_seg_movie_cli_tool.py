from faiaccpb_flatten_and_infer_and_compose_camera_pose_based import (
     faiaccpb_flatten_and_infer_and_compose_camera_pose_based
)
from prii import (
     prii
)
from grirosffmi_get_ram_in_ram_out_segmenter_from_final_model_id import (
     grirosffmi_get_ram_in_ram_out_segmenter_from_final_model_id
)
import argparse
import textwrap
import os



def priflatseg_cli_tool():
    """
    See also priseg for showing the union of all the ad boards segmentations.
    """

    argp = argparse.ArgumentParser(
        description="show the segmentation inference for a model of flatten then infer type",
        usage=textwrap.dedent(
            """
            export m=slflatpatch30
            export c=slday8game1
            priseg 0
            # or, override the clip_id and final_model_id:
            # priseg -m slflatpatch30 -c slday3game1 0
            """
        )
    )

    argp.add_argument(
        "frame_indices",
        nargs="+",
        type=int,
        help="a list of frame indices to infer. The final_model_id is the environment variable m and the clip_id is the environment variable c",
    )

    argp.add_argument(
        "-m", "--final_model_id",
        type=str,
        default=None,
        help="which (flattened) model to use for the ad board segmentation",
    )

    argp.add_argument(
        "-c", "--clip_id",
        type=str,
        default=None,
        help="which (flattened) model to use for the ad board segmentation",
    )
    
    args = argp.parse_args()
    
    frame_indices = args.frame_indices
    final_model_id = args.final_model_id
    clip_id = args.clip_id


    if final_model_id is None:
        try:  
            final_model_id = os.environ["m"]
        except KeyError:
            raise ValueError(
                textwrap.dedent(
                    """
                    please set the environment variable m to the final_model_id like:
                    
                    export m=slflatpatch30
                    
                    or specify it via the flag -m / --final_model_id like:
                    
                    priflatseg -m slflatpatch30 1000
                    """
                )
            )
    if clip_id is None:
            
        try:
            clip_id = os.environ["c"]
        except KeyError:
            raise ValueError(
                textwrap.dedent(
                    """
                    please set the environment variable c to the clip_id like:
                    
                    export c=slday8game1
                    
                    or specify it via the flag -c / --clip_id like:
                    
                    priflatseg -m slflatpatch30 -c slgame9day1 1000
                    """
                )
            )
    

    assert (
        clip_id not in ["brewcub"]
    ), "This does not work for baseball, because it is camera-pose / lens distortion based."

    print(f"For clip_id {clip_id} and final_model_id {final_model_id}, we will show the segmentation for frame_indices {frame_indices}")
    ram_in_ram_out_segmenter = grirosffmi_get_ram_in_ram_out_segmenter_from_final_model_id(
        final_model_id=final_model_id
    )

    board_ids = [
        "board0",
    ]
    board_id_to_rip_height = {
         "board0": 256
    }
    
    board_id_rip_width = {
        "board0": 4268
    }

    for frame_index in frame_indices:
        composition_rgb_hwc_np_u8 = faiaccpb_flatten_and_infer_and_compose_camera_pose_based(
            ram_in_ram_out_segmenter=ram_in_ram_out_segmenter,
            clip_id=clip_id,
            frame_index=frame_index,
            board_ids=board_ids,
            board_id_to_rip_height=board_id_to_rip_height,
            board_id_rip_width=board_id_rip_width,
            return_rgba=False,
            verbose=False,
        )
        prii(composition_rgb_hwc_np_u8)

# find /shared/clips/slday8game1/flat/board0/4268x256/frames -name '*_original.png' | wc -l

# # edit it to say 63000 to 80000 for footlocker:
# nano ~/sl.json5

# bat ~/sl.json5

# export m=slflatpatch30

# export j=~/sl.json5

# time python ~/r/major_rewrite/ifatj_infer_flattened_according_to_json5.py $j

# ls -1 /shared/clips/slday8game1/flat/board0/4268x256/masks/slflatpatch30/

# ls -1 /shared/clips/slday8game1/flat/board0/4268x256/masks/slflatpatch30/ | wc -l    

# export m="slflatpatch30"
# export c="slday8game1"
# export a=80000
# export b=81000


# mev_make_evaluation_video \
# --frames_dir /shared/clips/slday8game1/flat/board0/4268x256/frames \
# --original_suffix "_original.png" \
# --masks_dir /shared/clips/slday8game1/flat/board0/4268x256/masks/slflatpatch30 \
# --clip_id $c \
# --first_frame_index $a \
# --last_frame_index $b \
# --model_id $m \
# --fps 59.94 \
# --fill_color green \
# --out_suffix flat