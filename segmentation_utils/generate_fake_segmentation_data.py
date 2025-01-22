"""
Given:
- RGBA frame and solved camera
- RGBA relevance mask (from solved camera)
- RGBA player mask (only really necessary overtop of the relevance mask)

Generate:
- New RGBA player mask (pasted players)
- New RGBA relevance mask (unoccluded lane OR pasted player(s) over lane)
    -> To clarify, the original player segmentations become "irrelevant" regions except where new players are pasted over them

This way we can mix and match players and floors.
"""

import better_json as bj
from PIL import Image
import sys
import numpy as np
import random
from pathlib import Path

def generate_fake_data(
    path_to_frame, 
    path_to_player_mask,
    path_to_relevance_mask,
    dir_of_player_cutouts,  # '~/r/gsw1/player_cutouts'
    min_num_new_players=1,
    max_num_new_players=5,
    ):

    """
    frame is just the original RGBA image
    player_mask is RGBA frame only with alpha == 0 where not a player ball, or ref.
    relevance_mask is RGBA frame with alpha == 0 where not relevant.
    """
    
    num_players = np.random.randint(min_num_new_players, max_num_new_players+1) # number of players/refs to paste over region

    frame_rgba = Image.open(str(Path(path_to_frame).expanduser())).convert("RGBA")
    player_mask_rgba = Image.open(str(Path(path_to_player_mask).expanduser())).convert("RGBA") # alpha = 0 where it is LANE and NOT PLAYER
    relevance_mask_rgba = Image.open(str(Path(path_to_relevance_mask).expanduser())).convert("RGBA") # alpha = 0 where it is LANE

    player_indicator = np.array(player_mask_rgba)[:,:,-1] != 0 # This is nonzero overr the players, zero elsewhere
    #relevance_indicator = np.array(relevance_mask_rgba)[:,:,-1] == 0  # if revelcance region is alpha==0
    relevance_indicator = np.array(relevance_mask_rgba)[:,:,-1] != 0 # This is nonzero over the relevant region, zero elsewhere
    if (relevance_indicator!=0).sum() < 0.01 * relevance_indicator.size:
        return None # don't bother if it's barely on the screen
    idxs = np.nonzero(relevance_indicator) # list of spots to place cutouts

    list_of_all_player_cutout_paths = []
    for p in Path(dir_of_player_cutouts).expanduser().iterdir():
        if p.is_file():
            list_of_all_player_cutout_paths.append(p)
    player_cutout_paths = random.sample(list_of_all_player_cutout_paths, num_players)

    new_player_mask_rgba = Image.new(size=(frame_rgba.width, frame_rgba.height), mode="RGBA")
    for player_path in player_cutout_paths:

        # load the player cutout and pick a random point to anchor to the lane
        player_cutout_rgba = Image.open(player_path).convert("RGBA")
        player_cutout_indicator = Image.fromarray(np.array(player_cutout_rgba)[:,:,-1])
        player_ij = [
            int(player_cutout_rgba.height * np.random.rand()),
            int(player_cutout_rgba.width * np.random.rand())
        ]

        # choose a random location in the lane to which to anchor the cutout player
        idx = random.randint(0, len(idxs[0]))
        lane_ij = [idxs[0][idx], idxs[1][idx]]

        # paste the player onto the original frame
        frame_rgba.paste(
            player_cutout_rgba, 
            (lane_ij[1]-player_ij[1], lane_ij[0]-player_ij[0]), 
            player_cutout_indicator
        )

        # add it to the new player mask
        new_player_mask_rgba.paste(
            player_cutout_rgba, 
            (lane_ij[1]-player_ij[1], lane_ij[0]-player_ij[0]), 
            player_cutout_indicator
        )

    # the new relevance mask excludes the original players except where they are overwritten by the new players
    new_player_indicator = np.array(new_player_mask_rgba)[:,:,-1]!=0
    new_relevance_mask = np.logical_or(
        np.logical_and(
            relevance_indicator,
            np.logical_not(player_indicator)
        ),
        np.logical_and(
            relevance_indicator,
            new_player_indicator
        )
    )
    new_relevance_mask_rgba = Image.fromarray(new_relevance_mask[:,:,None] *  frame_rgba)

    ans = dict(
        frame_rgba = frame_rgba,
        player_mask_rgba = new_player_mask_rgba,
        relevance_mask_rgba = new_relevance_mask_rgba,
    )
    return ans


if __name__ == "__main__":
    import os
    import subprocess
    from PIL import Image, ImageOps

    num_per_frame = 2
    min_num_new_players = 5
    max_num_new_players = 10

    suffix = "_players.png"
    dir_of_output = Path("~/awecom/data/clips/gsw1/fake_segmentation_data/").expanduser()
    for filename in os.listdir(Path("~/r/gsw1/segmentation").expanduser()):
        if filename.endswith(suffix):
            print(filename)
            for i in range(num_per_frame):
                print(i, end='\r')
                name = os.path.basename(os.path.basename(filename)[:-len(suffix)])
                path_to_player_mask = Path(f"~/r/gsw1/segmentation/{filename}")
                #path_to_frame = Path(f"~/awecom/data/clips/gsw1/frames/{name}.jpg").expanduser()
                path_to_frame = Path(f"~/r/gsw1/segmentation/{name}_color.png").expanduser()
                for side in ['left', 'right']:
                    path_to_relevance_mask = Path(f"~/r/gsw1/segmentation/{name}_{side}_lane.png").expanduser()

                    ans = generate_fake_data(
                        path_to_frame = path_to_frame, #"~/r/gsw1/segmentation/gsw1_223731_color.png", 
                        path_to_player_mask = path_to_player_mask, #"~/r/gsw1/segmentation/gsw1_223731_nonlane.png",
                        path_to_relevance_mask = path_to_relevance_mask, #"~/r/gsw1/segmentation/gsw1_223731_relevant_lane.png",
                        dir_of_player_cutouts = '~/r/gsw1/player_cutouts',
                        min_num_new_players=min_num_new_players,
                        max_num_new_players=max_num_new_players,
                    )
                    if ans is None:
                        continue

                    # save everything
                    ans["frame_rgba"].save( dir_of_output / f"{name}_{i}_frame.png" )
                    ans["player_mask_rgba"].save( dir_of_output / f"{name}_{i}_players.png" )
                    ans["relevance_mask_rgba"].save( dir_of_output / f"{name}_{i}_{side}_lane.png" )

                    # Flatten it
                    border = 2
                    y_max = 7.83 + border
                    y_min = -y_max
                    if side == "left":
                        x_min = -47 - border
                        x_max = -28.17 + border
                    elif side == "right":
                        x_min = 28.17 - border
                        x_max = 47 + border
                    path_to_camera = Path(f"~/awecom/data/clips/gsw1/tracking_attempts/fourth/{name}_camera_parameters.json").expanduser()
                    width = 720
                    subprocess.call( # for the frame
                        [
                            #Path("~/r/floor-ripper/build/bin/mask_floor.exe").expanduser(),
                            Path("~/r/floor-ripper/rip_floor.exe").expanduser(),
                            str(path_to_camera),
                            str(x_min),
                            str(x_max),
                            str(y_min),
                            str(y_max),
                            str(width),
                            str(dir_of_output / f"{name}_{i}_frame.png"),
                            str(dir_of_output / f"{name}_{i}_{side}_frame_flat.png")
                        ]
                    )
                    subprocess.call( # for the relevance mask
                        [
                            #Path("~/r/floor-ripper/build/bin/mask_floor.exe").expanduser(),
                            Path("~/r/floor-ripper/rip_floor.exe").expanduser(),
                            str(path_to_camera),
                            str(x_min),
                            str(x_max),
                            str(y_min),
                            str(y_max),
                            str(width),
                            str(dir_of_output / f"{name}_{i}_{side}_lane.png"),
                            str(dir_of_output / f"{name}_{i}_{side}_lane_flat.png")
                        ]
                    )
                    subprocess.call( # for the player mask
                        [
                            #Path("~/r/floor-ripper/build/bin/mask_floor.exe").expanduser(),
                            Path("~/r/floor-ripper/rip_floor.exe").expanduser(),
                            str(path_to_camera),
                            str(x_min),
                            str(x_max),
                            str(y_min),
                            str(y_max),
                            str(width),
                            str(dir_of_output / f"{name}_{i}_players.png"),
                            str(dir_of_output / f"{name}_{i}_{side}_players_flat.png")
                        ]
                    )
                    if side == "left": # reflect left-right
                        im = Image.open(dir_of_output / f"{name}_{i}_{side}_lane_flat.png")
                        im_mirror = ImageOps.mirror(im)
                        im_mirror.save(dir_of_output / f"{name}_{i}_{side}_lane_flat.png")

                        im = Image.open(dir_of_output / f"{name}_{i}_{side}_frame_flat.png")
                        im_mirror = ImageOps.mirror(im)
                        im_mirror.save(dir_of_output / f"{name}_{i}_{side}_frame_flat.png")

                        im = Image.open(dir_of_output / f"{name}_{i}_{side}_players_flat.png")
                        im_mirror = ImageOps.mirror(im)
                        im_mirror.save(dir_of_output / f"{name}_{i}_{side}_players_flat.png")

                    # add average to blank zones
                    avg = Image.open(Path("~/r/segmentation_utils/avg_rhs_lane.png").expanduser())
                    img = Image.open(str(dir_of_output / f"{name}_{i}_{side}_lane_flat.png"))
                    arr = np.array(img)
                    img.paste(avg, mask=Image.fromarray(arr[:,:,-1]==0))
                    #img.save(str(dir_of_output / f"{name}_{i}_{side}_lane_flat_new.png"))

                    im = Image.open(dir_of_output / f"{name}_{i}_{side}_players_flat.png")
                    img.paste(im, mask=Image.fromarray(np.array(im)[:,:,-1]!=0))
                    img.save(str(dir_of_output / f"{name}_{i}_{side}_lane_flat_new.png"))

        for filename in os.listdir(dir_of_output):
            if filename.endswith("new.png"):
                continue
            elif filename.endswith("players_flat.png"):
                continue
            os.remove(f"{(Path(dir_of_output) / filename).expanduser()}")