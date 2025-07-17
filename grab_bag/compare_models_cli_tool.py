from see_alpha_channel_differences import (
     see_alpha_channel_differences
)
from colorama import Fore, Style
from prii import (
     prii
)

from make_rgba_from_original_and_mask_paths import (
     make_rgba_from_original_and_mask_paths
)
import shutil
from write_rgba_hwc_np_u8_to_png import (
     write_rgba_hwc_np_u8_to_png
)
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
import argparse
from pathlib import Path
import textwrap
from infer_arbitrary_frames import (
     infer_arbitrary_frames
)


def compare_models_cli_tool():
    argp = argparse.ArgumentParser(
        description=textwrap.dedent(
            """\
            The old model had bad frames,
            and we want to see if the new model is any better on those frames.
            """
        ),
        usage=textwrap.dedent(
            """\
            rm ~/compare/*

            python compare_models_cli_tool.py \\
            --old mft \\
            --new td0 \\
            --prii \\
            --rgba_out_dir ~/compare
            """
        )
    )

    argp.add_argument(
        "--old",
        type=str,
        required=True,
        default=None,
    )

    argp.add_argument(
        "--new",
        type=str,
        required=True,
        default=None,
    )

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
    if rgba_out_dir is not None:
        assert rgba_out_dir.is_dir(), f"{rgba_out_dir=} is not an extant directory"

    old_model_id = opt.old
    new_model_id = opt.new
    final_model_ids = [old_model_id, new_model_id]

    directories_to_pull_bad_frames_from = [
        Path("~/r/maccabi_fine_tuning/.approved").expanduser().resolve(),
    ]
    for directory in directories_to_pull_bad_frames_from:
        assert directory.is_dir(), f"{directory} is not an extant directory"
    originals = []
    for directory in directories_to_pull_bad_frames_from:
        for p in directory.glob("*_original.jpg"):
            originals.append(p)
        for p in directory.glob("*_original.png"):
            originals.append(p)
        # TODO: record and display what the human said about the frame

    originals = sorted(originals)

    print(f"{Fore.YELLOW}We are going to compare the final_models {final_model_ids} on these originals:{Style.RESET_ALL}")
    for original in originals:
        print(f"{original=}")

    shared_dir = get_the_large_capacity_shared_directory()
    inferences_dir = shared_dir / "inferences"

    do_the_inferencing = True
    if do_the_inferencing:
        for final_model_id in [old_model_id, new_model_id]:
            list_of_input_and_output_file_paths = []
            for original in originals:
                annotation_id = original.name[:-len("_original.jpg")]
                mask_file_path = inferences_dir / f"{annotation_id}_{final_model_id}.png"
                list_of_input_and_output_file_paths.append(
                    (original, mask_file_path)
                )
            
            infer_arbitrary_frames(
                final_model_id=final_model_id,
                list_of_input_and_output_file_paths=list_of_input_and_output_file_paths,
            )

    # TODO: move this to the caller, the printing is too advanced
    if print_in_terminal:
        for original_path in originals:
            print(f"\n\n\n\noriginal image {original_path.name}:")
            prii(original_path)
            if rgba_out_dir is not None:
                shutil.copy(
                    src=original_path,
                    dst=rgba_out_dir
                )

            for final_model_id in [old_model_id, new_model_id]:
                annotation_id = original_path.name[:-len("_original.jpg")]
                mask_path = inferences_dir / f"{annotation_id}_{final_model_id}.png"

                print(f"model {final_model_id} says:")
                rgba = make_rgba_from_original_and_mask_paths(
                    original_path=original_path,
                    mask_path=mask_path,
                    flip_mask=False,
                    quantize=False,
                )
                prii(rgba)

                if rgba_out_dir is not None:
                    assert rgba_out_dir.is_dir(), f"{rgba_out_dir=} is not a directory"
                    # rgba_out_path = rgba_out_dir / f"{clip_id}_{i:06d}_{final_model_id}.png"
                    rgba_out_path = rgba_out_dir / mask_path.name
                    write_rgba_hwc_np_u8_to_png(
                        rgba_hwc_np_u8=rgba,
                        out_abs_file_path=rgba_out_path,
                        verbose=False,
                    )
                
            see_alpha_channel_differences(
                alpha_source_a_file_path=inferences_dir / f"{annotation_id}_{old_model_id}.png",
                alpha_source_b_file_path=inferences_dir / f"{annotation_id}_{new_model_id}.png",
                rgb_source_file_path=original_path,
                save_file_path=None,
                print_in_terminal=True,
            )

            
            



if __name__ == "__main__":
    compare_models_cli_tool()
