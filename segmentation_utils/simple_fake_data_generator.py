"""
Given:
- RGBA background image

Generrate:
- Players stretched and rotated, pasted over background
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

    min_N = int(sys.argv[1])
    max_N = int(sys.argv[2])
    min_num_new_players = 1
    max_num_new_players = 10
    min_stretch = 1
    max_stretch = 3

    output_dir = Path("~/awecom/data/clips/gsw1/fake_segmentation_data/").expanduser()
    for i in range(min_N, max_N):
        print(i, end='\r')
        ans = generate_fake_data(
            path_to_background_image = Path(f"~/r/segmentation_utils/avg_rhs_lane.png").expanduser(),
            player_cutout_dir = Path('~/r/gsw1/player_cutouts').expanduser(),
            min_num_new_players=min_num_new_players,
            max_num_new_players=max_num_new_players,
            min_stretch=min_stretch,
            max_stretch=max_stretch
        )

        # save everything
        ans["frame_rgba"].save( output_dir / f"{i}_frame.png" )
        ans["mask_rgba"].save( output_dir / f"{i}_players.png" )