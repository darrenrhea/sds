from get_clip_id_and_frame_index_from_file_name import (
     get_clip_id_and_frame_index_from_file_name
)
import shutil
from prii_rgb_and_alpha import (
     prii_rgb_and_alpha
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
import sys
from gbccmaeem_get_biggest_connected_component_mask_and_everything_else_mask import (
     gbccmaeem_get_biggest_connected_component_mask_and_everything_else_mask
)
import numpy as np
from open_mask_image import (
     open_mask_image
)
from pathlib import Path
from prii import (
     prii
)
from get_mask_path_from_original_path import (
     get_mask_path_from_original_path
)
from scipy.ndimage import binary_fill_holes


def blend_masks():

    bad_examples = [
        500,
        1000,
        4000,
        6000,
        10000,
        11000,
        13500,
        21000,
        21500,
        334500,
        338500,
        345500
    ]

   

   
    folder = Path(
        "~/r/nfl-59778-skycam_floor/sarah/unassigned"
    ).expanduser()

    # original_paths = [
    #     folder / f"nfl-59778-skycam_{e:06d}_original.jpg"
    #     for e in bad_examples
    # ]

    original_paths = sorted(
        [
            original_path
            for original_path in folder.glob("*_original.jpg")
        ]
    )

   

    out_folder = Path(
        "~/r/nfl-59778-skycam_floor/sarah/out"
    ).expanduser()
    out_folder.mkdir(exist_ok=True, parents=True)


    preannotations_dir = Path(
        "~/a/preannotations/floor_not_floor/nfl-59778-skycam"
    ).expanduser()

    for original_path in original_paths:
        _, frame_index = get_clip_id_and_frame_index_from_file_name(
                file_name=original_path.name
        )
        print(frame_index)

        shutil.copy(
            src=original_path,
            dst=out_folder / original_path.name
        )
        good_on_players_mask_path = get_mask_path_from_original_path(
            original_path=original_path
        )
        
        if frame_index in bad_examples:
            print("skipping", frame_index)
            shutil.copy(
                src=good_on_players_mask_path,
                dst=out_folder / good_on_players_mask_path.name
            )
            continue

       
        
        
        rgb = open_as_rgb_hwc_np_u8(
            original_path
        )

        good_on_boundary_mask_path = preannotations_dir / good_on_players_mask_path.name
        prii(good_on_boundary_mask_path)
        print(good_on_boundary_mask_path)
        # prii(good_on_players_mask_path, caption="\n\n\ngood_on_players_mask:")
        # prii(good_on_boundary_mask_path, caption="\n\n\ngood_on_boundary_mask:")
        
        good_on_players_alpha = open_mask_image(good_on_players_mask_path)
        good_on_boundary_alpha = open_mask_image(good_on_boundary_mask_path)

        binary_good_on_players = (good_on_players_alpha > 32).astype(np.uint8)
        binary_good_on_boundary = (good_on_boundary_alpha > 32).astype(np.uint8)
        # prii(255 * binary_good_on_players)

        (
            biggest_connected_component_good_on_players_mask,
            everything_else_mask
        ) = gbccmaeem_get_biggest_connected_component_mask_and_everything_else_mask(
            binary_hw_np_u8=binary_good_on_players
        )

        (
            biggest_connected_component_good_on_boundary_mask,
            _
        ) = gbccmaeem_get_biggest_connected_component_mask_and_everything_else_mask(
            binary_hw_np_u8=binary_good_on_boundary
        )

        # prii_rgb_and_alpha(
        #     rgb,
        #     255 *    biggest_connected_component_good_on_players_mask,
        #     caption="biggest_connected_component_good_on_players_mask"
        # )

        # prii_rgb_and_alpha(
        #     rgb,
        #     255 *    biggest_connected_component_good_on_boundary_mask,
        #     caption="biggest_connected_component_good_on_boundary_mask"
        # )
        
        # biggest_connected_component_good_on_players_no_holes_mask = binary_fill_holes(biggest_connected_component_good_on_players_mask).astype(np.uint8)
        # prii(255 * biggest_connected_component_good_on_players_mask, caption="biggest_connected_component_no_holes_mask")

        players = np.minimum(
            good_on_players_alpha, 255*(1-biggest_connected_component_good_on_players_mask)
        )

        boundary = np.minimum(
            good_on_boundary_alpha,
            255*biggest_connected_component_good_on_players_mask,
        )
        # prii(players, caption="players")
        # prii(boundary, caption="boundary")
        total_alpha = np.maximum(players, boundary)
        rgba = np.zeros(
            shape=(1080, 1920, 4),
            dtype=np.uint8,
        )
        rgba[:, :, 3] = total_alpha
        rgba[:, :, :3] = rgb
        out_file_path = out_folder / good_on_players_mask_path.name
        prii(rgba, caption="rgba", out=out_file_path)

    print(f"ff {out_folder}")
     

     

if __name__ == "__main__":
    blend_masks()