import sys
from pathlib import Path
import textwrap
import better_json as bj


def get_all_sets_of_frame_ad_2d_correspondences():
    """
    This function is a generator that yields all the frame-ad correspondences.
    """
    save_dir = Path(
        "~/r/keypoint_correspondences_data"
    ).expanduser()

    assert (
        save_dir.exists()
    ), textwrap.dedent(
        f"""\
        {save_dir=} does not exist, maybe you need to clone it?"
        (cd ~/r && git clone git@github.com:darrenrhea/keypoint_correspondences)
        """
    )
    all_sets = []
    for p in save_dir.glob("*.json"):
        jsonable = bj.load(p)
        try:
            clip_id, frame_index, ad_id = jsonable["clip_id"], jsonable["frame_index"], jsonable["ad_id"]
        except KeyError:
            print(f"Something is wrong with {p}")
            sys.exit(1)
        
        jsonable["abs_file_path_str"] = str(p.resolve())

        all_sets.append(jsonable)
    return all_sets


def get_all_sets_of_frame_ad_2d_correspondences_for_this_clip_id(clip_id):
    return [
        x for x in get_all_sets_of_frame_ad_2d_correspondences() if x["clip_id"] == clip_id
    ]


def get_all_sets_of_frame_ad_2d_correspondences_for_this_ad_id(ad_id):
    return [
        x for x in get_all_sets_of_frame_ad_2d_correspondences() if x["ad_id"] == ad_id
    ]

 