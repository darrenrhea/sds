from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)

import pprint as pp

from pathlib import Path

def decide_what_to_flatten():

    mother_dir = Path("/shared/fake_game5")
    subdir_names = [
        "24_BOS_Finals_CSS_v01",
        "2k25_summer_league_july_12_22_espn_espn2",
        "ABC_NBA_Draft_CS_BOS",
        "ABC_NBA_Finals_THURS_Game6_IfNec",
        "ABC_NHL_Stanley_Cup_Game5_CS_BOS",
        "bad_boys_ride_or_die_will_smith",
        "emirates_fly_better_emirates_fly_better",
        "Flag",
        "follow_every_golden_trophy_finals_moment_nba_app_black",
        "get_in_the_game_vote",
        "golden_different_here",
        "golden_trophy_tshirt_nba_store",
        "nba_draft_2024_presented_by_state_farm_begins_june_26",
        "NBA_Finals_THURS_Game6_IfNec_CS_BOS",
        "state_farm_state_farm",
        "youtube_tv_youtube_tv",
        "ripbased2",
        "human_annotated_multiplied",
    ]
    new_mother_dir = Path("/shared/flattened_fake_game5")
    new_mother_dir.mkdir(exist_ok=True, parents=True)


    list_of_dicts = []
    for subdir_name in subdir_names:
        src_dir = mother_dir / subdir_name
        dst_dir = new_mother_dir / subdir_name
        dst_dir.mkdir(exist_ok=True, parents=True)

        original_file_paths = (
            list(src_dir.glob("*_original.png"))
            +
            list(src_dir.glob("*_original.jpg"))
        )
        for original_file_path in original_file_paths:
            assert original_file_path.is_file()
            prefix = original_file_path.name[:-len("_original.jpg")]
            if "fake" in prefix:
               clip_id_underscore_frame_index = original_file_path.name[:-len("_fake710839147215981_original.png")]
            elif "anothercopy" in prefix:
                temp = original_file_path.name[:-len("_original.jpg")]
                clip_id_underscore_frame_index = temp[len("anothercopy-02-"):]
            else:
                raise ValueError(f"{prefix=}")
            print(f"{prefix=}")
            print(f"{clip_id_underscore_frame_index=}")
            clip_id = clip_id_underscore_frame_index[:-7]
            print(f"{clip_id=}")
            frame_index = int(clip_id_underscore_frame_index[-6:])
            print(f"{frame_index=}")

            relevance_name = prefix + "_relevance.png"
            nonfloor_name = prefix + "_nonfloor.png"
            
            relevance_file_path = original_file_path.with_name(relevance_name)
            mask_file_path = original_file_path.with_name(nonfloor_name)
            assert relevance_file_path.is_file()
            assert mask_file_path.is_file()
         
            camera_pose = get_camera_pose_from_clip_id_and_frame_index(
                clip_id=clip_id,
                frame_index=frame_index,
            )
            
            save_original_file_path = dst_dir / original_file_path.name
            save_relevance_file_path = dst_dir / relevance_file_path.name
            save_mask_file_path = dst_dir / mask_file_path.name

            record = dict(
                clip_id=clip_id,
                frame_index=frame_index,
                camera_pose=camera_pose,
                original_file_path=original_file_path,
                relevance_file_path=relevance_file_path,
                mask_file_path=mask_file_path,
                save_original_file_path=save_original_file_path,
                save_relevance_file_path=save_relevance_file_path,
                save_mask_file_path=save_mask_file_path,
            )
            list_of_dicts.append(record)
            pp.pprint(record)
    return list_of_dicts
      
if __name__ == "__main__":
    list_of_dicts = decide_what_to_flatten()
    pp.pprint(list_of_dicts)