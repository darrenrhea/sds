"""
We are given in the folder ~/needs_color_correction/<context_id>/

1. A floor rip of this basketball court named <context_id>_ripped.png

2. and a bunch of extracted video frames, also showing portions of that same court

3. a list of color names in color_names.json

This program will iterate through the color_names,
for each color telling the user the color name,
and having them 

1. click on that color multiple times in the floor-rip
2. click on the color multiple times each video frame to the extent it is visible.

When the color is not visible in a video frame the user should simply press Spacebar.
"""

import sys
from colorama import Fore, Style
from pathlib import Path
from print_image_in_iterm2 import print_image_in_iterm2
from finitely_many_clicks_on_one_image import finitely_many_clicks_on_one_image
import better_json as bj
from sample_color_from_one_image import sample_color_from_one_image
from get_average_color_from_samples import get_average_color_from_samples
from valid_context_ids import valid_context_ids


context_id = sys.argv[1]
assert context_id in valid_context_ids

data_dir = Path("~/needs_color_correction").expanduser() / context_id
assert data_dir.is_dir()

color_names_file_path = data_dir / f"{context_id}_color_names.json"
all_color_names = bj.load(color_names_file_path)

floor_rip_file_path = data_dir / f"{context_id}_ripped.png"
assert floor_rip_file_path.is_file()

video_frame_file_paths = []
for p in data_dir.glob("*.jpg"):
    print(p.name)
    video_frame_file_paths.append(p)

mutating_output_file_path = data_dir / f"{context_id}_color_map.json"
if mutating_output_file_path.is_file():
    mutating_output = bj.load(mutating_output_file_path)
else:
    mutating_output = []

colors_to_not_do_again = [
    dct["color_name"]
    for dct in mutating_output
]

color_names = [
    color_name
    for color_name in all_color_names
    if color_name not in colors_to_not_do_again
]

for color_name in color_names:
    image_id_to_samples = dict()

    print(
        f"Starting sampling process for the color named {color_name}"
    )

    instructions = f"click several times on {color_name} if it is visible then spacebar"

    # First, we sample the color from points in the floor_rip:
    floor_rip_samples = sample_color_from_one_image(
        image_file_path=floor_rip_file_path,
        instructions=instructions
    )
    image_id_to_samples[floor_rip_file_path.name] = floor_rip_samples
    average_floor_rip_color = get_average_color_from_samples(floor_rip_samples)

    all_video_frame_samples = []
    for video_frame_file_path in video_frame_file_paths:
        # sample the color from points in the video_frame if possible:
        video_frame_samples = sample_color_from_one_image(
            image_file_path=video_frame_file_path,
            instructions=instructions
        )
        image_id_to_samples[video_frame_file_path.name] = video_frame_samples
        all_video_frame_samples += video_frame_samples
    
    average_video_frame_color = get_average_color_from_samples(all_video_frame_samples)
    
    print(f"Apparently the color {average_floor_rip_color} should map to {average_video_frame_color}")

    if mutating_output_file_path.is_file():
        mutating_output = bj.load(mutating_output_file_path)
    else:
        mutating_output = []
    
    mutating_output.append(
        dict(
            color_name=color_name,
            domain=average_floor_rip_color,
            range=average_video_frame_color,
            image_id_to_samples=image_id_to_samples
        )
    )

    bj.save(fp=mutating_output_file_path, obj=mutating_output)




