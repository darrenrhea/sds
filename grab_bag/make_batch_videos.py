"""
This is crap and you should not use it.
See flatten_leds/new_make_batch_videos.py for a better thing.
"""
from pathlib import Path
import subprocess
import better_json as bj

shared_dir = "/shared"
# clip_id = "bos-dal-2024-06-06-srt"
# clip_id = "bos-mia-2024-04-21-mxf"

# model_id = "npt"
# model_id = "nam"
# model_id = "edl"
# model_id = "nhl"
# model_id = "nid"
# model_id = "ndh"
# model_id = "enf"
# model_id = "eap"
# model_id = "emi"
# model_id = "nst"
# model_id = "ppn"
# model_id = "pbc"
# model_id = "gen"
# model_id = "e65"
# model_id = "r30"
# model_id = "r195"
# model_id = "g235"
# model_id = "rip"
model_id = "n1f"

ranges_dir = Path("~/r/frame_attributes").expanduser()
json_name = f"ad_ranges_{clip_id}.json5"
ranges_json_path = ranges_dir / json_name
print(f"Getting ranges from {ranges_json_path}")
ad_to_ranges = bj.load(ranges_json_path)

show_n_tell_path = Path(shared_dir).expanduser() / "show_n_tell"
show_n_tell_clips_path = show_n_tell_path / f"{model_id}"
show_n_tell_clips_path.mkdir(exist_ok=True)

select_ads = [
    # "24_BOS_Finals_CSS_v01",
    # "Draft_Awareness_CS_BOS",
    "YTTV_CS_BOS",
    # "Playoffs_Title_CS_BOS",
    # "NBA_APP_MSFT_CS_BOS",
    # "ESPN_DAL_LAC_NEXT_ABC_CS_BOS",
    # "NHL_Playoffs_CS_BOS",
    # "NBA_ID_CS_BOS",
    # "different_here",
    # "ESPN_NBA_Finals_CS_BOS",
    # "ESPN_APP_CS_BOS",
    # "ESPN_MIL_IND_FRI_CS_BOS",
    # "NBA_Store_CS_BOS",
    # "Playoffs_PHI_NYK_TOM_TNT_CS_BOS",
    # "PickEm_Bracket_Challenge_CS_BOS",
]

def make_video_command_list(shared_dir, clip_id, a, b, model_id, ad, view_type, fillcolor=None):
    if fillcolor is not None and view_type == "foreground":
        args = [
            "time",
            "mev_make_evaluation_video",
            "--original_suffix",
            "_original.jpg",
            "--frames_dir",
            f"{shared_dir}/clips/{clip_id}/frames",
            "--masks_dir",
            f"{shared_dir}/inferences",
            "--clip_id",
            f"{clip_id}",
            "--first_frame_index",
            f"{a}",
            "--last_frame_index",
            f"{b}",
            "--model_id",
            f"{model_id}",
            "--fps",
            "50.0",
            "--out_suffix",
            f"{ad}",
            "--fill_color",
            f"{fillcolor}",
            "--subdir",
            f"{model_id}"
        ]
    elif view_type == "background":
        args = [
                "time",
                "mev_make_evaluation_video",
                "--original_suffix",
                "_original.jpg",
                "--frames_dir",
                f"{shared_dir}/clips/{clip_id}/frames",
                "--masks_dir",
                f"{shared_dir}/inferences",
                "--clip_id",
                f"{clip_id}",
                "--first_frame_index",
                f"{a}",
                "--last_frame_index",
                f"{b}",
                "--model_id",
                f"{model_id}",
                "--fps",
                "50.0",
                "--background",
                "--out_suffix",
                f"{ad}",
                "--subdir",
                f"{model_id}"
            ]
    return args

for ad, pairs in ad_to_ranges.items():
    if ad in select_ads:
        for pair in pairs:
            a = pair[0]
            b = pair[1]
            args = make_video_command_list(
                shared_dir=shared_dir, 
                clip_id=clip_id, 
                a=a,
                b=b, 
                model_id=model_id, 
                ad=ad, 
                view_type="foreground", 
                fillcolor="green")
            subprocess.run(args, cwd="/home/drhea/r/major_rewrite")
            args = make_video_command_list(
                shared_dir=shared_dir, 
                clip_id=clip_id, 
                a=a,
                b=b, 
                model_id=model_id, 
                ad=ad, 
                view_type="foreground", 
                fillcolor="black")
            subprocess.run(args, cwd="/home/drhea/r/major_rewrite")
            args = make_video_command_list(
                shared_dir=shared_dir, 
                clip_id=clip_id, 
                a=a,
                b=b, 
                model_id=model_id, 
                ad=ad, 
                view_type="background", 
                fillcolor=None)
            subprocess.run(args, cwd="/home/drhea/r/major_rewrite")