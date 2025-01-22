"""
Given:
- RGBA frame and solved camera
- RGBA player mask (only really necessary overtop of the relevant region)

Do:
- Paste stretched/rotated player cutouts over lane
- Rip lane to canonical form
- Fill in old playerers with lane average

This way we can mix and match players and floors.
"""

import better_json as bj
from PIL import Image
import sys
import numpy as np
import random
from pathlib import Path

if __name__ == "__main__":
    import os
    import subprocess
    from PIL import Image, ImageOps
    from scipy import ndimage

    num_per_image = int(sys.argv[1])
    min_num_new_players = 1
    max_num_new_players = 5
    min_stretch = 0.75
    max_stretch = 1.25

    pad = 2
    x_bnds = dict(
        left = [-47-pad, -28.17+pad],
        right = [28.17-pad, 47+pad]
    )
    y_min = -7.83 - pad
    y_max = 7.83 + pad

    output_width = 720 # ripped lane image widt
    path_to_background_image = Path(f"~/r/segmentation_utils/avg_rhs_lane.png")

    list_of_all_player_cutout_paths = []
    player_cutout_dir = Path('~/r/gsw1/player_cutouts')
    for p in Path(player_cutout_dir).expanduser().iterdir():
        if p.is_file():
            list_of_all_player_cutout_paths.append(p)

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

        path_to_frame = Path(f"~/r/gsw1/segmentation/{name}_color.png")
        path_to_camera = Path(f"~/awecom/data/clips/gsw1/tracking_attempts/second/{name}_camera_parameters.json")
        path_to_player_mask = player_mask_dir / filename 

        img = Image.open(path_to_frame.expanduser()).convert("RGBA")
        mask = Image.open(path_to_player_mask.expanduser()).convert("RGBA")

        # dilate the mask
        num_dilation_iters = 7
        mask_arr = np.array(mask)[:,:,-1] # pull out alpha channel
        dilated_mask_arr = ndimage.binary_dilation(mask_arr!=0, iterations=num_dilation_iters)
        #dilated_mask_img = ImageOps.invert( Image.fromarray(dilated_mask_arr).convert("L") )
        dilated_mask_img = Image.fromarray(dilated_mask_arr).convert("L")
        mask = dilated_mask_img.convert("L")

        img.putalpha(ImageOps.invert(mask))
        #img.putalpha(Image.fromarray(np.array(mask)[:,:,-1]==0).convert("L"))
        tmp_path = "./tmp/noplayers.png"
        img.save(tmp_path)

        for side in ["left", "right"]:
            x_min, x_max = x_bnds[side]
            lane_mask_path = "./tmp/lane_mask.png"
            exec_string = "~/r/floor-ripper/build/bin/mask_floor.exe %s %s %s %s %s %s %s %s" % (path_to_camera, img.width, img.height, x_min, x_max, y_min, y_max, lane_mask_path)
            #exec_string = "~/r/floor-ripper/mask_floor.exe %s %s %s %s %s %s %s %s" % (path_to_camera, img.width, img.height, x_min, x_max, y_min, y_max, lane_mask_path)
            os.system(exec_string)
            lane_indicator = np.array(Image.open(lane_mask_path))[:,:,-1]
            if lane_indicator.sum() == 0:
                continue
            idxs = np.nonzero(lane_indicator) # list of lane pixel locations

            for i in range(num_per_image):
                background_rgba = img.copy()
                num_players = np.random.randint(min_num_new_players, max_num_new_players+1) # number of players/refs to paste over region
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

                    # choose a random location in the lane to which to anchor the cutout player
                    idx = random.randint(0, len(idxs[0]))
                    lane_ij = [idxs[0][idx], idxs[1][idx]]

                    # paste the player onto the original frame
                    background_rgba.paste(
                        player_cutout_rgba, 
                        (lane_ij[1]-player_ij[1], lane_ij[0]-player_ij[0]), 
                        mask=player_cutout_rgba
                    )
                    path_to_background_rgba = Path("./tmp/backkground_rgba.png").expanduser()
                    background_rgba.save(path_to_background_rgba)

                    # add it to the new player mask
                    mask_rgba.paste(
                        player_cutout_rgba, 
                        (lane_ij[1]-player_ij[1], lane_ij[0]-player_ij[0]), 
                        mask=player_cutout_rgba
                    )
                    path_to_mask_rgba = Path("./tmp/mask_rgba.png").expanduser()
                    mask_rgba.save(path_to_mask_rgba)

                output_path = "./tmp/ripped_frame.png"
                #exec_string = "~/r/floor-ripper/rip_floor.exe %s %s %s %s %s %s %s %s" % (path_to_camera, x_min, x_max, y_min, y_max, output_width, path_to_background_rgba, output_path)
                exec_string = "~/r/floor-ripper/build/bin/rip_floor.exe %s %s %s %s %s %s %s %s" % (path_to_camera, x_min, x_max, y_min, y_max, output_width, path_to_background_rgba, output_path)
                os.system(exec_string)

                #output_path = "./tmp/ripped_players.png"
                player_path = Path(f"~/awecom/data/clips/gsw1/fake_segmentation_data/{cnt}_players.png").expanduser()
                #exec_string = "~/r/floor-ripper/rip_floor.exe %s %s %s %s %s %s %s %s" % (path_to_camera, x_min, x_max, y_min, y_max, output_width, path_to_mask_rgba, player_path)
                exec_string = "~/r/floor-ripper/build/bin/rip_floor.exe %s %s %s %s %s %s %s %s" % (path_to_camera, x_min, x_max, y_min, y_max, output_width, path_to_mask_rgba, player_path)
                os.system(exec_string)

                # fill in frame with average
                frame = Image.open(output_path)
                players = Image.open(player_path)
                if side == "left":
                    frame = ImageOps.mirror(frame)
                    players = ImageOps.mirror(players)
                    players.save(player_path)
                path_to_background_image = Path(f"~/r/segmentation_utils/avg_rhs_lane.png")
                bg = Image.open(path_to_background_image.expanduser())
                mask = Image.fromarray(np.array(frame)[:,:,-1]==0).convert("L")
                frame.paste(bg, mask=mask)
                path_to_new_bg = Path(f"~/awecom/data/clips/gsw1/fake_segmentation_data/{cnt}_frame.png").expanduser()
                frame.save(path_to_new_bg)
                cnt += 1




"""

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
                """
