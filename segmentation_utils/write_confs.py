"""
run this to place a bunch of config files is subdir c
Then

"""
import better_json as bj
from pathlib import Path


clip_ids = [
    "BKN_CITY_2021-11-16_PGM_short",
    "BKN_CITY_2021-11-17_PGM_short",
    "BKN_CITY_2021-11-03_PGM_short",
    # "BKN_CITY_2022-01-07_PGM_short",
    # "BKN_CITY_2022-01-09_PGM_short",
    # "BKN_CITY_2022-01-26_PGM_short",
    # "BKN_CITY_2021-11-27_PGM_short",
    # "BKN_CITY_2021-11-30_PGM_short",
    # "BKN_CITY_2021-12-03_PGM_short",
    # "BKN_CITY_2021-12-04_PGM_short",
    # "BKN_CITY_2021-12-16_PGM_short",
    # "BKN_CITY_2021-12-18_PGM_short",
    # "BKN_CITY_2021-12-30_PGM_short",
    # "BKN_CITY_2022-01-01_PGM_short",
    # "BKN_CITY_2022-01-03_PGM_short", 
]

gpu_substring_which_gpu_pairs = [
    ("A5000", 0),
    ("A5000", 1),
    ("8000", 0),
]

Path("c").mkdir(exist_ok=True)

for index, clip_id in enumerate(clip_ids):
    gpu_substring, which_gpu = gpu_substring_which_gpu_pairs[index % 3]
    jsonable = {
        "clip_id": clip_id,
        "first_frame_index": 0,
        "last_frame_index": 14400,
        "gpu_substring": gpu_substring,
        "which_gpu": which_gpu,
        "architecture": "resnet34",
        "nn_input_width": 320,
        "nn_input_height": 280,
        "original_width": 1920,
        "original_height": 1080,
        "downsample_factor": 1,
        "model_name": "brooklyn_320p_280p_res34_32e_88f_downsampled_one_half_lam",
        "masking_attempt_id": "brooklyn_320p_280p_res34_32e_88f_downsampled_one_half_lam_bw",
        "increment_frame_index_by": 1,
        "save_color_information_into_masks": False
    }
    bj.dump(obj=jsonable, fp=f"c/{clip_id}.jsonc")
