from pathlib import Path
import subprocess
import sys
import better_json as bj

clip_id = "bos-dal-2024-06-06-srt"
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
# model_id = "gre"
# model_id = "e65"
# model_id = "r30"
# model_id = "r195"
# model_id = "g235"
# model_id = "rip"
model_id = "n1f"

ranges_dir = Path("~/r/frame_attributes").expanduser()
json_name = f"ad_ranges_{clip_id}.json5"
ranges_json_path = ranges_dir / json_name
print(ranges_json_path)

ad_to_ranges = bj.load(ranges_json_path)

select_ads = [
    # "24_BOS_Finals_CSS_v01",
    # "Draft_Awareness_CS_BOS",
    "YTTV_CS_BOS"
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

for ad, ranges in ad_to_ranges.items():
    if ad in select_ads:
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