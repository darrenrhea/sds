import numpy as np
from pathlib import Path
from homographically_warp_advertisement import homographically_warp_advertisement
import PIL.Image

if __name__ == "__main__":
    ads_dir = Path("~/r/led_ads").expanduser().resolve()
    ad_image_path = ads_dir / "statefarm.jpg"
    ad_image = np.array(PIL.Image.open(str(ad_image_path)))

    points_in_dst = np.array(
        [
            [  27.601727, 266.95654 ],
            [  28.19188,  349.9616  ],
            [ 218.11534,  349.97012 ],
            [ 217.6036,   266.9688  ],
        ]
    )
    
    warped_ad_image = homographically_warp_advertisement(
        image_np=ad_image,
        tl_bl_br_tr=points_in_dst[:4, :],
        out_width=1280,
        out_height=720
    )
    warped_ad_path = Path("warped_ad.png").resolve()
    PIL.Image.fromarray(warped_ad_image).save(str(warped_ad_path))
    print(f"pri {warped_ad_path}")


