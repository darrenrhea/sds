from get_video_frame_path_from_clip_id_and_frame_index import (
     get_video_frame_path_from_clip_id_and_frame_index
)
import shutil
from pathlib import Path
from write_rgba_hwc_np_u8_to_png import (
     write_rgba_hwc_np_u8_to_png
)
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
            export output_dir=~/preannotations
            export c=slday8game1
            priflatseg -c slday3game1 0
            priflatseg 0
            # or, override the clip_id and final_model_id:
            # priflatseg -m slflatpatch30 -c slday3game1 0
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

    argp.add_argument(
        "-o", "--output_dir",
        type=str,
        default=None,
        help="a directory to save the output images",
    )

    
    args = argp.parse_args()
    
    frame_indices = args.frame_indices
    final_model_id = args.final_model_id
    clip_id = args.clip_id
    output_dir = args.output_dir

    if output_dir is None:
        output_dir = os.environ.get("output_dir")
        if output_dir is not None:
            output_dir = Path(output_dir).resolve()
            assert output_dir.is_dir(), f"{output_dir} is not a directory"
    else:
        output_dir = Path(output_dir).resolve()
        assert output_dir.is_dir(), f"{output_dir} is not a directory"

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

    print(f"For {clip_id=} and {final_model_id=}, we will show the segmentation for frame_indices {frame_indices}")
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
        print(f"frame_index: {frame_index}")
        composition_rgba_hwc_np_u8 = faiaccpb_flatten_and_infer_and_compose_camera_pose_based(
            ram_in_ram_out_segmenter=ram_in_ram_out_segmenter,
            clip_id=clip_id,
            frame_index=frame_index,
            board_ids=board_ids,
            board_id_to_rip_height=board_id_to_rip_height,
            board_id_rip_width=board_id_rip_width,
            verbose=False,
            return_rgba=True,
        )
        
        if output_dir is not None:
            assert output_dir.is_dir(), f"{output_dir} is not an extant directory"
            sub_dir = output_dir / clip_id
            sub_dir.mkdir(exist_ok=True, parents=False)

            out_abs_file_path = sub_dir / f"{clip_id}_{frame_index:06d}_nonfloor.png"
            out_original_file_path = sub_dir / f"{clip_id}_{frame_index:06d}_original.jpg"
            write_rgba_hwc_np_u8_to_png(
                composition_rgba_hwc_np_u8,
                out_abs_file_path=out_abs_file_path,
                verbose=True
            )
            original_image_path = get_video_frame_path_from_clip_id_and_frame_index(
                clip_id=clip_id,
                frame_index=frame_index
            )
            shutil.copy(
                src=original_image_path,
                dst=out_original_file_path
            )
        else:
            print("You did not give an output directory, so we will show the output images")
            prii(composition_rgba_hwc_np_u8)


        
