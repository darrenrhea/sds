"""
Given:
- RGBA frame and solved camera
- RGBA player mask (only really necessary overtop of the relevant region)

Generate:
- Canonical image with masked players filled in with average

This way we can mix and match players and floors.
"""

import better_json as bj
from PIL import Image
import sys
import numpy as np
import random
from pathlib import Path

def generate_fake_data(
    path_to_background_image,
    player_cutout_dir,
    min_num_new_players=1,
    max_num_new_players=7,
    min_stretch=1,
    max_stretch=3
):
    num_players = np.random.randint(min_num_new_players, max_num_new_players+1) # number of players/refs to paste over region

    background_rgba = Image.open(str(Path(path_to_background_image).expanduser())).convert("RGBA")

    list_of_all_player_cutout_paths = []
    for p in Path(player_cutout_dir).expanduser().iterdir():
        if p.is_file():
            list_of_all_player_cutout_paths.append(p)
    player_cutout_paths = random.sample(list_of_all_player_cutout_paths, num_players)

    mask_rgba = Image.new(size=(background_rgba.width, background_rgba.height), mode="RGBA")
    for player_path in player_cutout_paths:
        # load the player cutout and pick a random point to anchor to the lane
        player_cutout_rgba = Image.open(player_path).convert("RGBA")
        stretch = min_stretch + max_stretch * np.random.rand()
        player_cutout_rgba = player_cutout_rgba.resize( 
            ( int(stretch * player_cutout_rgba.width), int(stretch * player_cutout_rgba.height) ), 
            Image.LANCZOS 
        ).rotate( 360 * np.random.rand(), expand=True )

        player_ij = [ # anchor point on player
            int(player_cutout_rgba.height * np.random.rand()),
            int(player_cutout_rgba.width * np.random.rand())
        ]

        lane_ij = [
            np.random.randint(0, background_rgba.height),
            np.random.randint(0, background_rgba.width)
        ]

        # paste the player onto the original frame
        background_rgba.paste(
            player_cutout_rgba, 
            (lane_ij[1]-player_ij[1], lane_ij[0]-player_ij[0]), 
            mask=player_cutout_rgba
        )

        # add it to the new player mask
        mask_rgba.paste(
            player_cutout_rgba, 
            (lane_ij[1]-player_ij[1], lane_ij[0]-player_ij[0]), 
            mask=player_cutout_rgba
        )

    ans = dict(
        frame_rgba = background_rgba,
        mask_rgba = mask_rgba,
    )
    return ans


if __name__ == "__main__":
    import os
    import subprocess
    from PIL import Image, ImageOps
    from scipy import ndimage

    num_per_image = int(sys.argv[1])
    min_num_new_players = 1
    max_num_new_players = 5
    min_stretch = 1
    max_stretch = 3

    pad = 2
    x_bnds = dict(
        left = [-47-pad, -28.17+pad],
        right = [28.17-pad, 47+pad]
    )
    y_min = -7.83 - pad
    y_max = 7.83 + pad

    output_width = 720 # ripped lane image widt
    path_to_background_image = Path(f"~/r/segmentation_utils/avg_rhs_lane.png")

    os.system("mkdir tmp")
    output_dir = Path("~/awecom/data/clips/gsw1/fake_segmentation_data/")
    player_mask_dir = Path("~/r/gsw1/segmentation")
    player_mask_suffix_0 = "_players.png"
    player_mask_suffix_1 = "_nonlane.png"
    cnt = 0
    for filename in os.listdir(player_mask_dir.expanduser()):
        if filename.endswith(player_mask_suffix_0):
            name = os.path.basename(os.path.basename(filename)[:-len(player_mask_suffix_0)])
        elif filename.endswith(player_mask_suffix_1):
            name = os.path.basename(os.path.basename(filename)[:-len(player_mask_suffix_1)])
        else:
            continue
        print(name)
        for i in range(num_per_image):
            path_to_frame = Path(f"~/r/gsw1/segmentation/{name}_color.png")
            path_to_camera = Path(f"~/awecom/data/clips/gsw1/tracking_attempts/second/{name}_camera_parameters.json")
            path_to_player_mask = player_mask_dir / filename 

            img = Image.open(path_to_frame.expanduser()).convert("RGBA")
            mask = Image.open(path_to_player_mask.expanduser()).convert("RGBA")
            img.putalpha(Image.fromarray(np.array(mask)[:,:,-1]==0).convert("L"))
            tmp_path = "./tmp/noplayers.png"
            img.save(tmp_path)

            for side in ["left", "right"]:
                x_min, x_max = x_bnds[side]
                output_path = "./tmp/background.png"
                exec_string = "~/r/floor-ripper/rip_floor.exe %s %s %s %s %s %s %s %s" % (path_to_camera, x_min, x_max, y_min, y_max, output_width, tmp_path, output_path)
                os.system(exec_string)

                img = Image.open(output_path)
                if side == "left":
                    img = ImageOps.mirror(img)
                bg = Image.open(path_to_background_image.expanduser())
                mask = Image.fromarray(np.array(img)[:,:,-1]==0).convert("L")
                if np.sum(np.array(mask)!=0) > 0.9 * np.prod(mask.size):
                    print(f"{side} lane not visible enough! skipping this side.")
                    continue

                num_dilation_iters = 7
                mask_arr = np.array(mask)#[:,:,-1] # pull out alpha channel
                dilated_mask_arr = ndimage.binary_dilation(mask_arr>0, iterations=num_dilation_iters)
                #dilated_mask_img = ImageOps.invert( Image.fromarray(dilated_mask_arr).convert("L") )
                dilated_mask_img = Image.fromarray(dilated_mask_arr).convert("L")
                mask = dilated_mask_img.convert("L")

                img.paste(bg, mask=mask)
                path_to_new_bg = Path("./tmp/background.png")
                img.save(path_to_new_bg)

                ans = generate_fake_data(
                    path_to_background_image = path_to_new_bg,
                    player_cutout_dir = Path('~/r/gsw1/player_cutouts').expanduser(),
                    min_num_new_players=min_num_new_players,
                    max_num_new_players=max_num_new_players,
                    min_stretch=min_stretch,
                    max_stretch=max_stretch
                )

                # save everything
                ans["frame_rgba"].save( (output_dir / f"{cnt}_frame.png").expanduser() )
                ans["mask_rgba"].save( (output_dir / f"{cnt}_players.png").expanduser() )
                cnt += 1