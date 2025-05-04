import textwrap
from get_original_path import (
     get_original_path
)
import shutil
from write_rgba_hwc_np_u8_to_png import (
     write_rgba_hwc_np_u8_to_png
)
from pathlib import Path
from prii import prii
from make_rgba_from_original_and_mask_paths import (
     make_rgba_from_original_and_mask_paths
)
from make_frame_ranges_file import (
     make_frame_ranges_file
)
from infer_from_id import (
     infer_from_id
)
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
from typing import List, Tuple, Union, Optional


def infer_arbitrary_frames_from_a_clip(
    final_model_id: str,
    clip_id: str,
    original_suffix: str,
    frame_ranges: List[
        Union[
            int,  # for a single frame
            Tuple[int, int],  # for a range of frames
            Tuple[int, int, int],  # for a range of frames with a step
        ],
    ],
    model_id_or_other_suffix: str,
    print_in_terminal: bool = False,
    rgba_out_dir: Optional[Path] = None,
):
    """
    A better thing than this is probably infer_arbitrary_frames.py
    You can either give a list of input and output file paths,
    Given a final_model_id,
    A clip_id THAT IS BLOWN OUT ENOUGH,
    and frame_ranges,
    which is a python list of single frames, or a frame_range [start, stop],
    or even a frame range with a step [start, stop, step],
    infer the frames from that clip under that model.
    """
    if rgba_out_dir is not None:
        assert (
            rgba_out_dir.is_dir()
        ), textwrap.dedent(
            f"""\
                {rgba_out_dir=}
                is not a directory, and we dont make directories.
                Suggest you make it yourself then try again.
            """
        )

    frame_ranges_file_path = make_frame_ranges_file(
        clip_id=clip_id,
        original_suffix=original_suffix,
        frame_ranges=frame_ranges
    )


    # shared_dir = get_the_large_capacity_shared_directory()
    shared_dir = "/shared"
    output_dir = shared_dir / "inferences"

    infer_from_id(
        final_model_id=final_model_id,
        model_id_suffix=final_model_id,
        frame_ranges_file_path=frame_ranges_file_path,
        output_dir=output_dir
    )
  
    # TODO: move this to the caller:

    indices = []
    for frame_range in frame_ranges:
        if isinstance(frame_range, int):
            print(f"inferred frame {frame_range}")
            indices.append(frame_range)
        elif len(frame_range) == 2:
            start, stop = frame_range
            print(f"inferred frames {start} to {stop}")
            indices.extend(range(start, stop+1))
        elif len(frame_range) == 3:
            start, stop, step = frame_range
            print(f"inferred frames {start} to {stop} with step {step}")
            indices.extend(range(start, stop+1, step))
        else:
            raise Exception(f"frame_range {frame_range} has length {len(frame_range)}")


    for i in indices:
        original_path = get_original_path(
            clip_id=clip_id,
            frame_index=i
        )
        if rgba_out_dir is not None or print_in_terminal:
            rgba = make_rgba_from_original_and_mask_paths(
                original_path=original_path,
                mask_path=output_dir / f"{clip_id}_{i:06d}_{final_model_id}.png",
                flip_mask=False,
                quantize=False,
            )
        if rgba_out_dir is not None:
            rgba_out_path = rgba_out_dir / f"{clip_id}_{i:06d}{model_id_or_other_suffix}.png"
            write_rgba_hwc_np_u8_to_png(
                rgba_hwc_np_u8=rgba,
                out_abs_file_path=rgba_out_path,
                verbose=False,
            )
            print(f"saved {rgba_out_path}")
            shutil.copy(
                original_path,
                rgba_out_dir
            )
            
        if print_in_terminal:
            print(f"model {final_model_id} on frame {original_path}")
            prii(original_path)
            prii(rgba)
