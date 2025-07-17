from pathlib import Path
import subprocess
import better_json as bj

ad = "skweek"
clip_id = "bay-zal-2024-03-15-mxf-yadif"
model_id = "skw"

# clip_id = "bay-efs-2023-12-20-mxf-yadif"
# model_id = "ana"



ad_to_ranges_json_path = Path(
    f"~/r/frame_attributes/ad_to_ranges_for_{clip_id}.json5"
).expanduser()

ad_to_ranges = bj.load(ad_to_ranges_json_path)

ranges = ad_to_ranges[ad]


for a, b in ranges:
    args = [
        "time",
        "python",
        "infer_cli_tool.py",
        "--final_model_id",
        f"{model_id}",
        "--clip_id",
        f"{clip_id}",
        "--original_suffix",
        "_original.jpg",
        "--start",
        f"{a}",
        "--end",
        f"{b}",
        "--step",
        "1"
    ]
    subprocess.run(args, cwd="/home/drhea/r/major_rewrite")