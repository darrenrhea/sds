from print_yellow import (
     print_yellow
)
from print_red import (
     print_red
)
import shutil
from get_metric_distance_between_two_image_paths import (
     get_metric_distance_between_two_image_paths
)

from extract_a_segment_of_frames_from_video import (
     extract_a_segment_of_frames_from_video
)
from print_green import (
     print_green
)
import sys
from extract_single_frame_from_video import (
     extract_single_frame_from_video
)
from get_clip_id_and_frame_index_from_original_file_name import (
     get_clip_id_and_frame_index_from_original_file_name
)
from pathlib import Path

from prii import prii


#   "stade       /hd2/s3/awecomai-temp/bal/fus-aia-2025-04-05.ts      267500 270800"
#   "southafrica /hd2/s3/awecomai-mxf-dropbox/BAL2024_SOUTHAFRICA.mxf 134000 138000"
#   "egypt       /hd2/s3/awecomai-mxf-dropbox/BAL2024_EGYPT.mxf       135400 138400"

# TODO: repos may have frames from several clip_ids / video files.  This assumes mono-clip_id
task_descriptions = dict(
    rwanda=dict(
        input_video_abs_file_path_str="/hd2/s3/awecomai-mxf-dropbox/BAL2024_UNKNOWN.mxf",  # weird exception for rwanda
        ann_repo_checkout_dir_str="~/r/bal2024_rwanda_floor",
        clip_id="bal2024_rwanda",
        fps=59.94, # TODO: use mediainfo or something to determine this
        deinterlace=False,
    ),
    senegal=dict(
        input_video_abs_file_path_str="/hd2/s3/awecomai-mxf-dropbox/BAL2024_SENEGAL.mxf",
        ann_repo_checkout_dir_str="~/r/bal2024_senegal_floor",
        clip_id="bal2024_senegal",
        fps=59.94, # TODO: use mediainfo or something to determine this
        deinterlace=False,
    ),
    southafrica=dict(
        input_video_abs_file_path_str="/hd2/s3/awecomai-mxf-dropbox/BAL2024_SOUTHAFRICA.mxf",
        ann_repo_checkout_dir_str="~/r/bal2024_southafrica_floor",
        clip_id="bal2024_southafrica",
        fps=59.94, # TODO: use mediainfo or something to determine this
        deinterlace=False,
    ),
    bal_rabat_20250410_aug=dict(
        color_correction_necessary=False,
        input_video_abs_file_path_str="/hd2/s3/awecomai-test-videos/nba/Mathieu/video_bal/bal_rabat_20250410_aug.mp4",
        ann_repo_checkout_dir_str="~/r/bal_rabat_20250410_aug_floor",
        clip_id="bal_rabat_20250410_aug",
        fps=50.00, # TODO: use mediainfo or something to determine this
        deinterlace=False,
    ),
    m1=dict(
        color_correction_necessary=False,
        input_video_abs_file_path_str="/hd2/s3/awecomai-original-videos/bal/20250412/bal_rabat2_00:00:00_10min_noaum.ts",
        ann_repo_checkout_dir_str="~/r/bal_rabat_floor",
        clip_id="bal_rabat_20250412",
        fps=59.94, # TODO: use mediainfo or something to determine this
        deinterlace=False,
    ),
    m2=dict(
        input_video_abs_file_path_str="/hd2/s3/awecomai-original-videos/bal/20250412/bal_rabat2_00:10:00_10min_noaum.mp4",
        ann_repo_checkout_dir_str="~/r/bal_rabat_20250412_floor",
        clip_id="bal_rabat_20250412_2",
        fps=50.0, # TODO: use mediainfo or something to determine this
        deinterlace=False,
    ),
    m3=dict(
        input_video_abs_file_path_str="/hd2/s3/awecomai-original-videos/bal/20250412/bal_rabat2_00:20:00_10min_noaum.mp4",
        ann_repo_checkout_dir_str="~/r/bal_rabat_20250412_floor",
        clip_id="bal_rabat_20250412_3",
        fps=59.94, # TODO: use mediainfo or something to determine this
        deinterlace=False,
    ),
    m4=dict(
        input_video_abs_file_path_str="/hd2/s3/awecomai-original-videos/bal/20250412/bal_rabat2_00:30:00_10min_noaum.mp4",
        ann_repo_checkout_dir_str="~/r/bal_rabat_20250412_floor",
        clip_id="bal_rabat_20250412_4",
        fps=59.94, # TODO: use mediainfo or something to determine this
        deinterlace=False,
    ),
    m5=dict(
        input_video_abs_file_path_str="/hd2/s3/awecomai-original-videos/bal/20250412/bal_rabat2_00:40:00_10min_noaum.mp4",
        ann_repo_checkout_dir_str="~/r/bal_rabat_20250412_floor",
        clip_id="bal_rabat_20250412_5",
        fps=59.94, # TODO: use mediainfo or something to determine this
        deinterlace=False,
    ),
    m6=dict(
        input_video_abs_file_path_str="/hd2/s3/awecomai-original-videos/bal/20250412/bal_rabat2_00:50:00_10min_noaum.mp4",
        ann_repo_checkout_dir_str="~/r/bal_rabat_20250412_floor",
        clip_id="bal_rabat_20250412_6",
        fps=59.94, # TODO: use mediainfo or something to determine this
        deinterlace=False,
    ),
     m7=dict(
        input_video_abs_file_path_str="/hd2/s3/awecomai-original-videos/bal/20250412/bal_rabat2_01:00:00_10min_noaum.mp4",
        ann_repo_checkout_dir_str="~/r/bal_rabat_20250412_floor",
        clip_id="bal_rabat_20250412_7",
        fps=59.94, # TODO: use mediainfo or something to determine this
        deinterlace=False,
    ),

    
    
    
    # fusaia=dict(
    #     input_video_abs_file_path_str="/hd2/s3/awecomai-temp/bal/fus-aia-2025-04-05.mp4",  # does it hate .ts or h265?
    #     ann_repo_checkout_dir_str="~/r/fus-aia-2025-04-05_floor",
    #     clip_id="fus-aia-2025-04-05",
    #     fps=50.0, # TODO: use mediainfo or something to determine this
    #     deinterlace=False,
    # ),
    # bal_game2_bigzoom=dict(
    #     input_video_abs_file_path_str="/hd2/clips/bal_game2_bigzoom/bal_game2_bigzoom.mp4",
    #     ann_repo_checkout_dir_str="~/r/bal_game2_bigzoom_floor",
    #     clip_id="bal_game2_bigzoom",
    #     fps=59.94, # TODO: use mediainfo or something to determine this
    #     deinterlace=False,
    # ),
)
png_or_jpg = "jpg"
pix_fmt = "yuvj422p"
task_description = task_descriptions["m2"]

# BEGIN unpack the task description:
clip_id = task_description["clip_id"]
input_video_abs_file_path_str = task_description["input_video_abs_file_path_str"]
ann_repo_checkout_dir_str = task_description["ann_repo_checkout_dir_str"]
fps = task_description["fps"]
deinterlace = task_description["deinterlace"]
# ENDOF unpack the task description.


input_video_abs_file_path = Path(input_video_abs_file_path_str).resolve()
assert input_video_abs_file_path.suffix in [".mp4", ".mxf"], f"ERROR: {input_video_abs_file_path} is not a .mp4 or .mxf file!"
if not input_video_abs_file_path.exists():
    print_red(f"ERROR: {input_video_abs_file_path} does not exist!")
    print_yellow("Suggest you do")
    print_green(
        f"ffmpeg -i {input_video_abs_file_path.parent / input_video_abs_file_path.stem}.ts  -c:v copy  {input_video_abs_file_path}"
    )
    sys.exit(1)

ann_repo_checkout_dir = Path(ann_repo_checkout_dir_str).expanduser()
assert ann_repo_checkout_dir.is_dir(), f"ERROR: {ann_repo_checkout_dir} is not an extant directory!"




original_paths = [
    p
    for p in ann_repo_checkout_dir.rglob("*_original.jpg")
]

pairs_of_frame_index_and_abs_out_path = []
for p in original_paths:
    file_name = p.name
    infered_clip_id, frame_index = get_clip_id_and_frame_index_from_original_file_name(
        file_name=file_name
    )
    if infered_clip_id != clip_id:
        continue
    pairs_of_frame_index_and_abs_out_path.append(
        (
            frame_index,
            p,
        )
    )
# print("[")
# for frame_index, out_frame_abs_file_path in pairs_of_frame_index_and_abs_out_path:
#     print(f'    ({frame_index}, "{out_frame_abs_file_path}"),')
# print("]")
# sys.exit(0)


for frame_index_to_extract, image_we_are_searching_for_file_path in pairs_of_frame_index_and_abs_out_path:
    print(f"Fixing color in {image_we_are_searching_for_file_path}")
    out_dir = Path("~/found").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_frame_abs_file_path = out_dir / image_we_are_searching_for_file_path.name

    print(f"frame_index_to_extract={frame_index_to_extract}")
    prii(image_we_are_searching_for_file_path, caption="what we are looking for:")
    temp_dir =  Path("~/temp").expanduser()
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    candidate_paths = extract_a_segment_of_frames_from_video(
        input_video_abs_file_path=input_video_abs_file_path,
        first_frame=frame_index_to_extract - 120,
        last_frame=frame_index_to_extract + 120,
        out_dir_abs_path=temp_dir,
        clip_id=clip_id,
        fps=fps,
        verbose=False,
    )
    argmin = None
    min_dist = float("inf")
    for candidate_path in candidate_paths:
        # prii(candidate_path)
        dist = get_metric_distance_between_two_image_paths(
            a_path=image_we_are_searching_for_file_path,
            b_path=candidate_path,
        )
        if dist < min_dist:
            min_dist = dist
            argmin = candidate_path
            
    print(f"min_dist={min_dist}")
    print(f"argmin={argmin}")
    
    shutil.copy(
        src=argmin,
        dst=image_we_are_searching_for_file_path,
    )
    
    print_green(f"ls {temp_dir}")
    prii(argmin, caption="argmin")
        
    rel_path = image_we_are_searching_for_file_path.relative_to(Path.home())
    s = f"~/{rel_path}"
    print(f"rsync dl:{argmin} {s}")
    print(f"ff {s[:-12]}*")


