import pandas
import better_json as bj
import os
from pathlib import Path

clip_id = "bay-zal-2024-03-15-mxf-yadif"
df = pandas.read_csv('EB_23-24_R29_BAY-ZAL_adsQ1.csv')
start=0
step=1
ad_dict = {}
for i in range(0, len(df), step):   
    previous_index = i
    next_index = i+step
    previous_ad = df.iloc[previous_index, 4].replace("interwetten;","")
    next_ad = df.iloc[next_index, 4].replace("interwetten;","")
    if next_ad == 'END':
        break
    if previous_ad == 'MISC':
        continue
    else:
        start = i
        start_index = df.iloc[previous_index, 2]
        end_index = df.iloc[next_index, 2]
        if previous_ad not in ad_dict.keys():
            ad_dict[previous_ad] = []

        ad_dict[previous_ad].append([int(start_index), int(end_index)])
out_dir = Path(os.getcwd())/"ad_ranges"
out_dir.mkdir(exist_ok=True)
json_name = f"{clip_id}_test.json5"
out_json_path = out_dir / json_name
bj.dump(obj=ad_dict, fp=out_json_path)

