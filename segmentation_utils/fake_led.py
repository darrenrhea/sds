import cv2
import numpy as np
import sys
import better_json as bj
from pathlib import Path
import shutil

ads_dir = Path("~/r/led_ads").expanduser()

long_ad_paths = [
    ads_dir / p
    for p in [
        "azure_banner_long.jpg",
        "bkn_versus_celtics_long.jpg",
        "carmax_long.jpg",
        "celtics_long.jpg",
        "content_long.png",
        "francisco_long.jpg",
        "header_long.png",
        "hyundai_long.png",
        "michelob_ultra_long.jpg",
        "microsoft_365_long.jpg",
        "microsoft__long_gradient.jpg",
        "microsoft_azure_long.png",
        "microsoft_gray_long.jpg",
        "office_365_long.png",
        "playstation_long.jpg",
        "purple_long.png",
        "redblue_long.jpg",
        "rosa_long.jpg",
        "sportstravel_long.jpg",
        "zoom_long.png",
    ]
]

short_ad_paths = [
    ads_dir / p
    for p in [
        "2022_short.jpg",
        "blue_short.jpg",
        "boston_celtics_short.jpg",
        "carmax_short.png",
        "celtics_playoffs_short.jpg",
        "clippers_short.jpg",
        "crypto_short.jpg",
        "finals_short.png",
        "linkedin_short.jpg",
        "microsoft_short.png",
        "nba-thats-game_short.jpg",
        "nba_playoffs_logo_short.jpg",
        "nba_short.jpg",
        "playoffs_google_short.jpg",
        "playoffs_short.png",
        "purple_short.jpg",
        "red_short.jpg",
        "sprint_short.jpg",
        "tmobile_short.jpg",
        "twitter_short.jpg",
        "whole_short.jpg",
        "youtube_short.png",
    ]
]

screen_name_to_list_of_possible_ad_paths = {
    "left_led": short_ad_paths,
    "right_led": short_ad_paths,
    "center_led": long_ad_paths,
    "left_goal": short_ad_paths,
    "right_goal": short_ad_paths
}


def warp_image(image, tl_bl_br_tr):
    ad_height = image.shape[0]
    ad_width = image.shape[1]

    src = np.array(
        [
            [0, 0],
            [0, ad_height],
            [ad_width, ad_height],
            [ad_width, 0],
        ],
        dtype=np.float32
    )
    dst = np.array(
        tl_bl_br_tr,
        dtype=np.float32
    )

    H = cv2.getPerspectiveTransform(src, dst)
    new_image = np.zeros(shape=(1080, 1920, 3), dtype=np.uint8)
    cv2.warpPerspective(src=image, dst=new_image, M=H, dsize=(1920, 1080))
    return new_image



def make_fake_led_image(
    nonled_path,
    json_path,
    screen_name_to_list_of_possible_ads,
    out_path
):
    """
    Given a RGBA PNG mask where the pixels inside the LED screens
    have been erased; and a JSON file describing for each
    LED screen the (x, y-growing-down)
    pixel locations of the corners of those screens
    Top Left, Bottom Left, Bottom Right Top Right;
    and a list of possible ads per screen (aspect ratio can make some
    ads inappropriate for a screen
    )
    Makes a synthetic / fake image where the LED screens have some
    random ads inserted into them.
    """
    # For the real deal make, opacity 1.0.
    # When debugging, 0.5 is nice to be able to see where things are:
    # opacity = 0.5
    opacity = 1.0
    nonled_mask_np_bgra_uint8 = cv2.imread(
        filename=str(nonled_path),
        flags=cv2.IMREAD_UNCHANGED  # without this, it will not load the alpha channel
    )

    assert (
        nonled_mask_np_bgra_uint8.shape[2] == 4
    ), f"ERROR: {nonled_path} is supposed to be an RGBA PNG, which would have 4 channels"

    alpha_uint8 = nonled_mask_np_bgra_uint8[:, :, 3]
    bgr_uint8 = nonled_mask_np_bgra_uint8[:, :, :3]
    foreground_bgr_float = bgr_uint8.astype(float)


    jsonable = bj.load(json_path)
    screen_name_to_tl_bl_br_tr = jsonable["screen_name_to_tl_bl_br_tr"]
    
    # We may be insertint several fake ads,
    # so we accumulate their renderings here:
    total_ads = np.zeros(shape=(1080, 1920, 3), dtype=np.uint8)

    for screen_name in screen_name_to_tl_bl_br_tr:
        tl_bl_br_tr = screen_name_to_tl_bl_br_tr[screen_name]
        list_of_possible_ads = screen_name_to_list_of_possible_ads[screen_name]
        num_ads = len(list_of_possible_ads)
        ad_path = list_of_possible_ads[np.random.randint(0, num_ads)]
        # print(ad_path)
        ad_np_bgr_uint8 = cv2.imread(
            filename=str(ad_path)
        )
        ad_np_bgr_uint8[:,:,:] = ad_np_bgr_uint8[:,:,:] // 4
        ad_rendering = warp_image(
            image=ad_np_bgr_uint8,
            tl_bl_br_tr=tl_bl_br_tr
        )

        total_ads = np.maximum(total_ads, ad_rendering)

    total_ads_float = total_ads.astype(float)

    # Normalize the alpha_uint8 mask to keep intensity between 0 and 1
    alpha_float = alpha_uint8.astype(float) / 255.0 * opacity
    # Multiply the foreground with the alpha_uint8 matte

    a = alpha_float[..., np.newaxis] * foreground_bgr_float
    # Multiply the background with ( 1 - alpha_uint8 )

    b = (1.0 - alpha_float[..., np.newaxis]) * total_ads_float
    composition = a + b
    
    cv2.imwrite(
        filename=str(out_path),
        img=composition
    )
    print(f"pri {out_path} &&")

def main():
    image_ids = [
        f"BOS_CORE_2022-03-30_MIA_PGM_{n:06d}"
        for n in [
            0,
            1000,
            2000,
            4000,
            17000,
            18000,
            19000,
            21000,
            22000,
            23000,
            27000,
        ]

    ]

    for image_id in image_ids:
        nonled_path = f"/home/drhea/r/boston_celtics/led/{image_id}_nonled.png"
        json_path = f"/home/drhea/r/boston_celtics/led/{image_id}.json"
        for k in range(1):
            rando = np.random.randint(0, 10**8)
            new_image_id = f"fake_{rando:08d}_{image_id}"
            out_path = f"/home/drhea/r/boston_celtics/fake_led/{new_image_id}.jpg"
            
            out_png_mask_path = f"/home/drhea/r/boston_celtics/fake_led/{new_image_id}_nonled.png"
            shutil.copy(nonled_path, out_png_mask_path)

            make_fake_led_image(
                nonled_path,
                json_path,
                screen_name_to_list_of_possible_ads=screen_name_to_list_of_possible_ad_paths,
                out_path=out_path
            )

if __name__ == "__main__":
    main()
