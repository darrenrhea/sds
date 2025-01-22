from pathlib import Path
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)

def get_ad_rgb_hwc_np_u8_by_ad_id(ad_id):
    ad_path = Path(
        f"~/r/nba_ads/summer_league_2024/{ad_id}.png"
    ).expanduser()

    ad = open_as_rgb_hwc_np_u8(ad_path)
    return ad
