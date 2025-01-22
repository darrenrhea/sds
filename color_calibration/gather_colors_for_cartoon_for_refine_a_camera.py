"""
We will eventually be assigning RGB values to Blender materials.

This program will iterate through the material_names,
for each color telling the user the material_name,
and having them 

1. click on that color multiple times in the professional photos
2. click on the color multiple times each video frame to the extent it is visible.

When the color is not visible in a video frame the user should simply press Spacebar.
"""


import numpy as np
import sys
from colorama import Fore, Style
from pathlib import Path
import better_json as bj
from sample_color_from_one_image import sample_color_from_one_image
from get_average_color_from_samples import get_average_color_from_samples
from rsync_utils import download_via_rsync
import pprint as pp
import subprocess


def interpret_response(
    response_str: str,
    colors_already_done_to_rgb_triplet: dict[str, list]
):
    """
    This either returns
        None, meaning the response was not interpretable;
        or
        the string "sample_via_clicks"
        or
        a Python list of length 3 filled with integers in the interval [0, 255]
    """
    
    print(f"Trying to interpret the response [{response_str}]")
    if response_str == "":
        return "sample_via_clicks"
    is_an_initial_segment_of_an_already_defined_color = False
    num_prefix_matches = 0
    average_color = None
    for color, rgb in colors_already_done_to_rgb_triplet.items():
        if response_str == color[:len(response_str)]:
            is_an_initial_segment_of_an_already_defined_color = True
            num_prefix_matches += 1
            average_color = rgb
    if num_prefix_matches >= 2:
        print("Too short a prefix: ambiguous")
        return None
    if is_an_initial_segment_of_an_already_defined_color:
        assert average_color is not None and len(average_color) == 3
        return average_color
    pieces = response_str.split(" ")
    if len(pieces) != 3:
        return None
    try:
        average_color = [int(p) for p in pieces]
        for k in range(3):
            assert 0 <= average_color[k] <= 255
        return average_color
    except:
        print(f"{Fore.RED}please enter a space-delimited triplet of RGB values in the range [0, 255]{Style.RESET_ALL}")
        return None



def get_average_color_via_clicking_program(color_name, professional_photo_file_paths):
    instructions = f"click several times on {color_name} if it is visible then spacebar"

    all_professional_photo_samples = []
    for professional_photo_file_path in professional_photo_file_paths:
        # sample the color from points in the professional_photo if possible:
        professional_photo_samples = sample_color_from_one_image(
            image_file_path=professional_photo_file_path,
            instructions=instructions
        )
        all_professional_photo_samples += professional_photo_samples
    
    rgb = get_average_color_from_samples(all_professional_photo_samples)
    
    print(f"Apparently {color_name} = {rgb}")

    assert isinstance(rgb, list)
    assert len(rgb) == 3
    for x in rgb:
        assert isinstance(x, int)
        assert 0 <= x and x <= 255
    
    return rgb

context_id = sys.argv[1]

professional_photo_selection_file_path = Path(
    "~/r/floor_modeling_tables/professional_photo_selection.json"
).expanduser()

professional_photo_selection = bj.load(
    professional_photo_selection_file_path
)

context = None
for entry in professional_photo_selection:
    if entry["context_id"] == context_id:
        context = entry
        break

assert (
    context is not None and context["context_id"] == context_id
), f"ERROR: Did you define photo_indices for {context_id} in {professional_photo_selection_file_path}?"

print(context)
assert (
    "photo_indices_for_color_sampling" in context
), f"ERROR: Did you define \"photo_indices_for_color_sampling\" for {context_id} in {professional_photo_selection_file_path}?"

photo_indices = entry["photo_indices_for_color_sampling"]


data_dir = Path(f"~/needs_color_correction").expanduser() / context_id
if not data_dir.exists():
    data_dir.mkdir(exist_ok=True, parents=True)
assert data_dir.is_dir()

professional_photos_dir = Path(
    f"~/tracker/registries/nba/photos/{context_id}"
).expanduser()


color_names_file_path = data_dir / f"professional_color_names.json"
if not color_names_file_path.exists():
    default_color_names = [
        "boundary_lines",
        "floor_no_apron_really",
        "key_aka_the_paint",
        "wood",
        "free_throw_half_circle",
        "lane_lines",
        "tpcirc",
        "sidemarks",
        "basemarks",
        "submarks",
        "lanemarks",
        "division_line",
        "circle_segments",
        "racirc",
    ]
    bj.dump(fp=color_names_file_path, obj=default_color_names)
all_color_names = bj.load(color_names_file_path)


professional_photo_file_paths = [
    professional_photos_dir / f"{context_id}_{k:04d}.jpg"
    for k in photo_indices
]

# popup the photo(s) in Preview:
for professional_photo_file_path in professional_photo_file_paths:
    args=["open", "-a", "preview", str(professional_photo_file_path)]
    print(" ".join(args))
    subprocess.run(args=args)

for index in photo_indices:
    do_download_via_rsync = True
    if do_download_via_rsync:
        download_via_rsync(
            src_machine="plumbus",
            src_path=f"/Volumes/NBA/2022-2023_Season_Photos/Courts_Renamed/{context_id}/{context_id}_{index:04d}.jpg",
            dst_path=str(professional_photos_dir) + "/"
        )
    else:
        args = [
            "dvc_pull",
            f"{context_id}_{index:04d}.jpg"
        ]
        subprocess.run(
            args=args,
            cwd=str(
                Path(f"~/tracker/registries/nba/photos/{context_id}").expanduser()
            )
        )


mutating_output_file_path = data_dir / f"professional_photograph_colors.json"
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

colors_already_done_to_rgb_triplet = dict()  # this is a dict[str, Annotated[List[int], 3]]
for color_name in color_names:
    while True:
        print(
            "\n\n\n\n\n\n\n"
            f"{Fore.YELLOW}What is the sRGB triplet for the color named {color_name}{Style.RESET_ALL}\n"
            f"Press [RETURN] to sample by clicking, or\ntype in a space-delimited triplet a la 240 255 129, or\nthe initial prefix of one of these already defined colors:"
        )
        for key, rgb in colors_already_done_to_rgb_triplet.items():
            print(f"{key}: {rgb}")
        
        response_str = input()
        interpretation = interpret_response(
            response_str=response_str,
            colors_already_done_to_rgb_triplet=colors_already_done_to_rgb_triplet
        )
        if interpretation is not None:  # None means
            break
        else:
            print("Response not understood, please try again")
    
    rgb = None  # later we will make sure it has been assigned a non-None value
    if interpretation == "sample_via_clicks":
        while True:
            rgb = get_average_color_via_clicking_program(
                color_name=color_name,
                professional_photo_file_paths=professional_photo_file_paths
            )
            response_str = input(
                f"Press RETURN if you happy with your answer for {color_name}, or any other response to do over"
            )
            if response_str == "":
                break

    elif isinstance(interpretation, list): # a write-in-candidate or a reference to an already defined value
        assert (
            len(interpretation) == 3
            and
            0 <= interpretation[0] <= 255
            and
            0 <= interpretation[1] <= 255
            and
            0 <= interpretation[2] <= 255
        )
        rgb = interpretation

   
    # store the answer
    colors_already_done_to_rgb_triplet[color_name] = rgb

    if mutating_output_file_path.is_file():
        mutating_output = bj.load(mutating_output_file_path)
    else:
        mutating_output = []
    
    mutating_output.append(
        dict(
            color_name=color_name,
            average=dict(
                r_average=rgb[0],
                g_average=rgb[1],
                b_average=rgb[2]
            )
        )
    )

    bj.save(fp=mutating_output_file_path, obj=mutating_output)



mutating_output = bj.load(mutating_output_file_path)
name_to_rgb = dict()

for item in mutating_output:
    color_name = item["color_name"]
    average = item["average"]
    r = int(average["r_average"])
    g = int(average["g_average"])
    b = int(average["b_average"])
    name_to_rgb[color_name] = [r, g, b]

print(f"for {context_id}, the colors are:")
for_blender = dict()
for color_name in all_color_names:
    r, g, b = name_to_rgb[color_name]
    rn = np.round(r / 255, 3)
    gn = np.round(g / 255, 3)
    bn = np.round(b / 255, 3)
    print(f"{color_name} = ({r / 255:.3f}, {g / 255:.3f}, {b / 255:.3f})\n\n")
    for_blender[color_name] = [rn, gn, bn]

# pp.pprint(for_blender)
