import subprocess
from pathlib import Path
import os
import better_json as bj


shared_dir = "/media/drhea/muchspace"
clip_id = "bay-zal-2024-03-15-mxf-yadif"
model_id = "bks"
# out_dir = Path(os.getcwd())/"ad_ranges"
# json_name = f"{clip_id}.json5"
ranges_dir = Path("~/r/frame_attributes").expanduser()
json_name = ranges_dir/f"ad_ranges_{clip_id}.json5"
ranges_json_path = ranges_dir / json_name
ad_json = bj.load(ranges_json_path)
show_n_tell_path = Path(shared_dir).expanduser() / "show_n_tell"
show_n_tell_clips_path = show_n_tell_path / f"{model_id}"
show_n_tell_concat_path = show_n_tell_path / f"{model_id}_concatenated"
show_n_tell_concat_path.mkdir(exist_ok=True)
concat_files_path = Path(os.getcwd()) / "clip_files"
concat_files_path.mkdir(exist_ok=True)
concat_file_combos = [
    ["foreground", "fillgreen"],
    ["foreground", "fillblack"],
    ["background", "fillblack"]
]

select_ads = [
    "bkt"
]

# Write the clips to concatenate to file, one file per ad.
# Associate each file write to a model id.
for k, v in ad_json.items():
    if k in select_ads:
        for combo in concat_file_combos:
            ad_file_list = concat_files_path / Path(f"{k}_{model_id}_{combo[0]}_{combo[1]}.txt").expanduser()
            if os.path.exists(ad_file_list):
                os.remove(ad_file_list)
            ad_file = open(ad_file_list, 'a')
            for range_tuple in v:
                video_clip_title = show_n_tell_clips_path / f"{clip_id}_from_{range_tuple[0]}_to_{range_tuple[1]}_{model_id}_{combo[0]}_{combo[1]}_{k}.mp4"
                ad_file.write(f"file {video_clip_title}\n")
            ad_file.close()
            full_video_filename = f"{k}_{clip_id}_{model_id}_{combo[0]}_{combo[1]}.mp4"

            full_out_path = show_n_tell_concat_path / full_video_filename
            print(f"Concatenating from {ad_file_list}")
            args = [
                "/usr/local/bin/ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                f"{ad_file_list}",
                "-c",
                "copy",
                f"{full_out_path}",
                "-y"
            ]
            subprocess.run(args, cwd="/home/drhea/r/major_rewrite")

            print(f"Concatenated video saved to {full_out_path}")