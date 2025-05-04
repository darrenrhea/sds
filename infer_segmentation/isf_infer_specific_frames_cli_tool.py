import argparse
from pathlib import Path
import textwrap
from infer_arbitrary_frames_from_a_clip import (
     infer_arbitrary_frames_from_a_clip
)

import better_json as bj

def isf_infer_specific_frames_cli_tool():
    argp = argparse.ArgumentParser(
        description=textwrap.dedent(
            """\
            Infer arbitrary frames from a clip,
            and for that matter put the grayscale inference together with the original image into preannotations / fixup-frame RGBA images
            and stuff them in a folder
            """
        ),
        usage=textwrap.dedent(
            """\
            
            export c=bay-zal-2024-03-15-mxf-yadif
            export m=u11

            isf_infer_specific_frames \\
            --final_model_id $m \\
            --clip_id $c \\
            --original_suffix _original.jpg \\
            ~/r/frame_attributes/${c}_led.json5 \\
            --prii \\
            --rgba_out_dir ~/temp

            Or for less verbose:

            isf_infer_specific_frames \\
            --final_model_id $m \\
            --clip_id $c \\
            --original_suffix _original.jpg \\
            ~/r/frame_attributes/${c}.json5 \\
            --rgba_out_dir ~/r/${c}

            
            """
        )
    )
    argp.add_argument(
        "--final_model_id",
        type=str,
        required=True,
        help="The final model id, like tw9"
    )
    argp.add_argument(
        "--clip_id",
        type=str,
        required=True,
        help="The clip id, like london20240208"
    )
    argp.add_argument(
        "--original_suffix",
        type=str,
        required=True,
        help="The original suffix, like .jpg or _original.jpg or _original.png or .png"
    )
    
    argp.add_argument(
        "frames_file",
        type=str,
        help="The file that lists which frames as a list of frame ranges to infer"
    ),

    argp.add_argument(
        "--rgba_out_dir",
        type=str,
        required=False,
        default=None,
    )
    argp.add_argument(
        '--prii',
        action='store_true'
    )

    opt =  argp.parse_args()
    print_in_terminal = opt.prii
    rgba_out_dir = Path(opt.rgba_out_dir).resolve() if opt.rgba_out_dir else None

    final_model_id = opt.final_model_id
    clip_id = opt.clip_id
    original_suffix = opt.original_suffix
    frames_file_path = Path(opt.frames_file).resolve()



    assert original_suffix in [".jpg", "_original.jpg", "_original.png", ".png", "_original_clahe.jpg"]


    frame_indices = bj.load(frames_file_path)
    assert isinstance(frame_indices, list), f"{frame_indices=} is not a list"
    for thing in frame_indices:
        assert isinstance(thing, int)

   
    
    infer_arbitrary_frames_from_a_clip(
        print_in_terminal=print_in_terminal,
        final_model_id=final_model_id,
        clip_id=clip_id,
        original_suffix=original_suffix,
        frame_ranges=frame_indices,
        rgba_out_dir=rgba_out_dir,
        model_id_or_other_suffix="_nonfloor"
    )


if __name__ == "__main__":
    isf_infer_specific_frames_cli_tool()
