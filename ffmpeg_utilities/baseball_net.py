from extract_single_frame_from_video import extract_single_frame_from_video

from pathlib import Path

def multiples_of_this_between_this_and_this(n, a, b):
    """
    Return a list of all the multiples of n between a and b (inclusive)
    """
    return [x for x in range(a, b + 1) if x % n == 0]

we_want_nets = False
if we_want_nets:
    # from baynzo:
    frame_ranges = [
        [480663, 481598],
        [482032, 482264],
        [484271, 487254],
        [491069, 491933],
        [496992, 497719],
        [498149, 498519],
    ]
else:
    # Some frame ranges from
    # ann_20200725PIT-STL-NC.mp4
    # where the shot resembles those with nets (but there isn't actually a net). the thinking is that these would be suitable for synthetic data where a fake net would be applied.
    frame_ranges = [
        [509002, 509424],
        [510986, 511724],
        [514462, 514843],
        [518032, 518369],
        [519395, 519892],
        [524233, 524436],
        [536298, 536565],
        [539895, 540543],
    ]

input_video_abs_file_path = Path(
    # "/Volumes/awecom/baseball/10_18_2021/ann_20200725PIT-STL-NC.mp4"
    "/Volumes/awecom/baseball/10_18_2021/20200725PIT-STL-NC.mp4"
)

clip_id = "STLvPIT_2020-07-25_NC_no_net"

out_dir = Path(
    f"~/baseball_frame_samples/{clip_id}"
).expanduser()

out_dir.mkdir(exist_ok=True, parents=True)

for frame_range in frame_ranges:
    first_index, last_index = frame_range
    frame_indices = multiples_of_this_between_this_and_this(
        n=100,
        a=first_index,
        b=last_index
    )

    for frame_index_to_extract in frame_indices:
        out_frame_abs_file_path = out_dir / f"{clip_id}_{frame_index_to_extract:06d}.jpg"

        extract_single_frame_from_video(
            input_video_abs_file_path=input_video_abs_file_path,
            frame_index_to_extract=frame_index_to_extract,
            out_frame_abs_file_path=out_frame_abs_file_path
        )

