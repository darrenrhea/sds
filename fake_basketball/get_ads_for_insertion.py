from pathlib import Path
from typing import List, Optional


def get_ads_for_insertion(
    white_list_for_ads_needing_color_correction: Optional[List[str]],
    white_list_for_ads_not_needing_color_correction: Optional[List[str]],
    ads_needing_color_correction_dir: Path,
    ads_not_needing_color_correction_dir: Path,
    just_a_few_ads_that_dont_need_color_correction=False,
    just_a_few_ads_that_do_need_color_correction=False,
):
    """
    This is what decides which ads get stuck into the video.
    """
    assert ads_not_needing_color_correction_dir.is_dir(), f"ERROR: {ads_not_needing_color_correction_dir=} is not a directory."
    assert ads_needing_color_correction_dir.is_dir(), f"ERROR: {ads_needing_color_correction_dir=} is not a directory."

    ad_name_to_paths_that_dont_need_color_correction = {}
    for p in ads_not_needing_color_correction_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name == ".git":
            continue
        ad_name = p.name
        if (
            white_list_for_ads_not_needing_color_correction is not None
            and
            ad_name not in white_list_for_ads_not_needing_color_correction
        ):
            continue
        ad_name_to_paths_that_dont_need_color_correction[ad_name] = [
            x for x in p.glob("*.png")
        ]

    ad_name_to_paths_that_do_need_color_correction = {}
    for p in ads_needing_color_correction_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name == ".git":
            continue
        ad_name = p.name
        if (
            white_list_for_ads_needing_color_correction is not None
            and
            ad_name not in white_list_for_ads_needing_color_correction
        ):
            continue
        ad_name_to_paths_that_do_need_color_correction[ad_name] = [
            x for x in sorted(p.glob("*.png"))
        ]

    if just_a_few_ads_that_dont_need_color_correction:
        ad_name_to_paths_that_dont_need_color_correction_str = {
            "skweek": [
                x
                for x in (ads_not_needing_color_correction_dir / "skweek").glob("*.png")
            ]
        }
        ad_name_to_paths_that_dont_need_color_correction = {
            ad_name: sorted([Path(x).expanduser() for x in ad_path_strs])
            for ad_name, ad_path_strs in ad_name_to_paths_that_dont_need_color_correction_str.items()
        }

    if just_a_few_ads_that_do_need_color_correction:
        ad_name_to_paths_that_do_need_color_correction_str = {
            "skweek": [
                x
                for x in (ads_needing_color_correction_dir / "skweek").glob("*.png")
            ]
        }
    
        ad_name_to_paths_that_do_need_color_correction = {
            ad_name: sorted([Path(x).expanduser() for x in ad_path_strs])
            for ad_name, ad_path_strs in ad_name_to_paths_that_do_need_color_correction_str.items()
        }

  
    return dict(
        ad_name_to_paths_that_do_need_color_correction=ad_name_to_paths_that_do_need_color_correction,
        ad_name_to_paths_that_dont_need_color_correction=ad_name_to_paths_that_dont_need_color_correction,
    )

   
if __name__ == "__main__":
    ads_needing_color_correction_dir = Path("~/r/ads_winnowed").expanduser()

    doesnt_need_color_correction_dir = Path("~/r/ads_that_dont_need_color_correction").expanduser()
   

    
    ans = get_ads_for_insertion(
        white_list_for_ads_needing_color_correction=None,  # None means all ads.
        ads_needing_color_correction_dir=ads_needing_color_correction_dir,
        ads_not_needing_color_correction_dir=doesnt_need_color_correction_dir,
        just_a_few_ads_that_dont_need_color_correction=False,
        just_a_few_ads_that_do_need_color_correction=False,
    )

    ad_name_to_paths_that_dont_need_color_correction = ans["ad_name_to_paths_that_dont_need_color_correction"]
    ad_name_to_paths_that_do_need_color_correction = ans["ad_name_to_paths_that_do_need_color_correction"]
    
    for ad_name, paths in ad_name_to_paths_that_dont_need_color_correction.items():
        print(f"{ad_name} that do not need color correction:")
        for p in paths:
            print(p)

    for ad_name, paths in ad_name_to_paths_that_do_need_color_correction.items():
        print(f"{ad_name} that do need color correction:")
        for p in paths:
            print(p)
    
    