from grirosffmi_get_ram_in_ram_out_segmenter_from_final_model_id import (
     grirosffmi_get_ram_in_ram_out_segmenter_from_final_model_id
)
from write_rgb_hwc_np_u8_to_png import (
     write_rgb_hwc_np_u8_to_png
)
from concatenate_videos import (
     concatenate_videos
)
import argparse
import textwrap
from instruct_how_to_download_file_to_laptop import (
     instruct_how_to_download_file_to_laptop
)
from make_plain_video import (
     make_plain_video
)
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
import os
from pathlib import Path
from faiac_flatten_and_infer_and_compose import (
     faiac_flatten_and_infer_and_compose
)
import better_json as bj



def iacfmffrij_infer_and_compose_flattened_model_for_frame_ranges_in_json5():
    """
    First, makes a bunch of contiguous frames videos,
    both composited and original.
    Then, concatenates all the composited into one giant composited video,
    and all the original videos into one giant original video.
    """

    argp = argparse.ArgumentParser(
        description=textwrap.dedent(
            """
            Make a original video and a composition video for each
            (clip_id, a, b) frame range triplet in the json5 file.
            """
        ),
        usage=textwrap.dedent(
            """
            export m=brewcubflattenedvip4
            python iacfmffrij_infer_and_compose_flattened_model_for_frame_ranges_in_json5.py ~/brewcub_short.json5
            """
        )
    )
    argp.add_argument(
        "json_file_path",
        help="The json5 file describing which (clip_id, a, b) frame range triplets to make movies for"
    )
    
   


    args = argp.parse_args()
    json_file_path = Path(args.json_file_path).resolve()
    
    assert json_file_path.exists(),f"{json_file_path=} does not exist"

    triplets = bj.load(json_file_path)
    assert isinstance(triplets, list), f"The JSON in:\n\n    {json_file_path}\nis not a list"

    for triplet in triplets:
        assert isinstance(triplet, list), f"{triplet=} is not a list"
        assert len(triplet) == 3, f"{triplet=} does not have 3 elements"
        clip_id = triplet[0]
        first_frame_index = triplet[1]
        last_frame_index = triplet[2]
        assert isinstance(clip_id, str), f"{clip_id=} is not a str"
        assert isinstance(first_frame_index, int), f"{first_frame_index=} is not an int"
        assert isinstance(last_frame_index, int), f"{last_frame_index=} is not an int"
        assert first_frame_index <= last_frame_index, f"{first_frame_index=} is not less than {last_frame_index=}"
    
    shared_dir = get_the_large_capacity_shared_directory()

    final_model_id = os.environ["m"]

    ram_in_ram_out_segmenter = (
        grirosffmi_get_ram_in_ram_out_segmenter_from_final_model_id(
            final_model_id=final_model_id
        )
    )

    out_video_file_paths = []
    original_out_video_file_paths = []

    for clip_id, first_frame_index, last_frame_index in triplets:
        composition_out_dir = Path(f"/shared/clips/{clip_id}/compositions")
        composition_out_dir.mkdir(exist_ok=True, parents=True)

        out_video_file_path = shared_dir / "show_n_tell" / f"{clip_id}_from_{first_frame_index}_to_{last_frame_index}_{final_model_id}.mp4"

        original_out_video_file_path = shared_dir / "show_n_tell" / f"{clip_id}_from_{first_frame_index}_to_{last_frame_index}.mp4"

        if out_video_file_path.exists():
            print(f"{out_video_file_path} already exists. Skipping.")
        else:      
            for frame_index in range(first_frame_index, last_frame_index + 1):
                composition_rgb_hwc_np_u8 = faiac_flatten_and_infer_and_compose(
                    ram_in_ram_out_segmenter=ram_in_ram_out_segmenter,
                    clip_id=clip_id,
                    frame_index=frame_index,
                    board_ids=["left", "right"],
                    board_id_to_rip_height={"left": 256, "right": 256},
                    board_id_rip_width={"left": 1024, "right": 1024},
                )

                out_abs_file_path = Path(
                    composition_out_dir / f"{clip_id}_{frame_index:06d}.png"
                ).expanduser().resolve()
                
                write_rgb_hwc_np_u8_to_png(
                    out_abs_file_path=out_abs_file_path,
                    rgb_hwc_np_u8=composition_rgb_hwc_np_u8,
                )

            make_plain_video(
                original_suffix=".png",
                frames_dir=Path(f"/shared/clips/{clip_id}/compositions"),
                first_frame_index=first_frame_index,
                last_frame_index=last_frame_index,
                clip_id=clip_id,
                fps=59.94,
                out_video_file_path=out_video_file_path,
            )

        if original_out_video_file_path.exists():
            print(f"{original_out_video_file_path} already exists. Skipping.")
        else:
            make_plain_video(
                original_suffix="_original.jpg",
                frames_dir=Path(f"/shared/clips/{clip_id}/frames"),
                first_frame_index=first_frame_index,
                last_frame_index=last_frame_index,
                clip_id=clip_id,
                fps=59.94,
                out_video_file_path=original_out_video_file_path,
            )

        instruct_how_to_download_file_to_laptop(
            file_path=out_video_file_path
        )

        instruct_how_to_download_file_to_laptop(
            file_path=original_out_video_file_path
        )

        out_video_file_paths.append(out_video_file_path)
        original_out_video_file_paths.append(original_out_video_file_path)

    concat_video_file_path = shared_dir / "show_n_tell" / f"{clip_id}_{final_model_id}.mp4"
    original_concat_video_file_path = shared_dir / "show_n_tell" / f"{clip_id}.mp4"

    concatenate_videos(
        video_abs_file_paths=out_video_file_paths,
        out_video_abs_file_path=concat_video_file_path,
        verbose=True
    )

    concatenate_videos(
        video_abs_file_paths=original_out_video_file_paths,
        out_video_abs_file_path=original_concat_video_file_path,
        verbose=True
    )

    print("\n\n\n\n\n\n\n\n\n\nHere is a repeat of all the downloadables:")

    for out_video_file_path in out_video_file_paths:
        instruct_how_to_download_file_to_laptop(
            file_path=out_video_file_path
        )

    instruct_how_to_download_file_to_laptop(
        file_path=concat_video_file_path
    )

    instruct_how_to_download_file_to_laptop(
        file_path=original_concat_video_file_path
    )


if __name__ == "__main__":
    iacfmffrij_infer_and_compose_flattened_model_for_frame_ranges_in_json5()
