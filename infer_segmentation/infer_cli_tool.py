import argparse
from pathlib import Path
import textwrap
from infer_arbitrary_frames_from_a_clip import (
     infer_arbitrary_frames_from_a_clip
)

# in argparse, the usage message is a format string,
# so percent signs must be escaped as double percent signs %%
infer_cli_tool_usage_message = textwrap.dedent(
    """\
    

    export c=bay-mta-2024-03-22-mxf
    export a=101800
    export b=102335
    export step=100
    export m=itw
    export original_suffix=_original.jpg
    
    export preannotations_dir=~/preannotations

    printf "export c=%%s\\n" $c &&
    printf "export m=%%s\\n" $m &&
    printf "export a=%%d\\n" $a  &&
    printf "export b=%%d\\n" $b  &&
    printf "export step=%%d\\n" $step &&
    printf "original_suffix=%%s\\n" $original_suffix

    printf "clip_id=%%s" $c
    printf "final_model_id=%%s" $m
    printf "first_frame_index=%%d" $a
    printf "last_frame_index=%%d" $b
    printf "step=%%d" $step
    printf "We expect the originals to have original_suffix=%%s" $original_suffix

   


    # run with no output other than the inferences
    python infer_cli_tool.py \\
    --final_model_id $m \\
    --clip_id $c \\
    --original_suffix $original_suffix \\
    --start $a \\
    --end $b \\
    --step $step

    python infer_cli_tool.py \\
    --final_model_id $m \\
    --clip_id $c \\
    --original_suffix $original_suffix \\
    --start $a \\
    --end $b \\
    --step $step \\
    --prii \\
    --rgba_out_dir ~/compare

    python infer_cli_tool.py \\
    --final_model_id $m \\
    --clip_id $c \\
    --original_suffix $original_suffix \\
    --start $a \\
    --end $b \\
    --step $step \\
    --prii \\
    --rgba_out_dir ~/compare

    # see it as it happens:
    python infer_cli_tool.py \\
    --final_model_id $m \\
    --clip_id $c \\
    --original_suffix $original_suffix \\
    --start $a \\
    --end $b \\
    --step $step \\
    --prii \\
    --rgba_out_dir $preannotations_dir
    """

)
def infer_cli_tool():
    argp = argparse.ArgumentParser(
        description="Infer arbitrary frames from a clip",
        usage=infer_cli_tool_usage_message,
    )
    argp.add_argument(
        "--final_model_id",
        type=str,
        required=True,
        help="The final model id, like tw9"
    )
    argp.add_argument(
        "--clip_mother_dir",
        type=str,
        required=False,
        default="/hd2/clips",
        help="The mother directory whose subdirectories contain blown out video frames, often /shared/clips or /hd2/clips"
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
        "--start",
        type=int,
        required=True,
        help="first frame"
    )
    argp.add_argument(
        "--end",
        type=int,
        required=True,
        help="last"
    )
    argp.add_argument(
        "--step",
        type=int,
        required=True,
        help="step"
    )
    argp.add_argument(
        "--rgba_out_dir",
        type=str,
        required=False,
        default=None,
    )
    argp.add_argument(
        '--out_suffix',
        type=str,
        required=False,
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
    first_frame_index = opt.start
    last_frame_index = opt.end
    step = opt.step
    out_suffix = opt.out_suffix
    clip_mother_dir = Path(opt.clip_mother_dir).resolve()
    
    if out_suffix is None:
        out_suffix = f"_{final_model_id}"

    assert original_suffix in [".jpg", "_original.jpg", "_original.png", ".png", "_original_clahe.jpg", "_original_clahe_t32.jpg"]

    infer_arbitrary_frames_from_a_clip(
        model_id_or_other_suffix=out_suffix,
        print_in_terminal=print_in_terminal,
        final_model_id=final_model_id,
        clip_id=clip_id,
        clip_mother_dir=clip_mother_dir,
        original_suffix=original_suffix,
        frame_ranges=[
           [first_frame_index, last_frame_index, step],
        ],
        rgba_out_dir=rgba_out_dir

    )


if __name__ == "__main__":
    infer_cli_tool()
