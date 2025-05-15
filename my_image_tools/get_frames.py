import sys
import subprocess
from pathlib import Path
from colorama import Fore, Style

usage_message = """
Yo this is a command-line utility to help people "self-serve" video frames to segment.

The current best videos,

    blend_first_half.mp4
    and
    blend_second_half.mp4

have green-on-black 6-digit numbers in the upper right corner, which we call
the canonical frame indices.

Enter your Slack name then one or more canonical frame indices like so to get alpha-channeled pngs in your folder todo folder:

    frames darren 149431

"""

readme_content = """
# Segmentation

Look in the "todo" folder for frames for you to fix up.

Work in the todo directory until you are satisfied with your mask.
As you make progress, keep exporting it as a PNG from GIMP with RGB info
as <your_folder>/todo/gsw1_123456_nonfloor.png with the correct frame number.

When you think it is Done, move it to your ready_for_review folder

    mv todo/gsw1_123456_nonfloor.png ready_for_review

    rsync ~/seg/<your_name>/ready_for_review/ lam:~/seg/<your_name>/ready_for_review/

"""

def main():
    if len(sys.argv) < 3:
        print(usage_message)
        sys.exit(1)
    
    person = sys.argv[1]
    assert person in [
        "anna",
        "cav",
        "chaz",
        "darren",
        "felix",
        "icee",
        "pat",
        "rachael",
        "tenzin",
    ]

    original_dir = Path("~/awecom/data/clips/gsw1/frames").expanduser()
    mask_dir = Path("~/awecom/data/clips/gsw1/masking_attempts/final_bw").expanduser()
    seg_dir = Path(f"~/seg").expanduser()
    seg_dir.mkdir(exist_ok=True)
    personal_dir = Path(f"~/seg/{person}").expanduser()
    personal_dir.mkdir(exist_ok=True)
    todo_dir = personal_dir / "todo"
    todo_dir.mkdir(exist_ok=True)
    review_dir = personal_dir / "ready_for_review"
    review_dir.mkdir(exist_ok=True)
    readme_path = personal_dir / "README.md"
    with open(readme_path, "w") as fp:
        fp.write(readme_content)
    

    frame_indices = []
    for k in range(2, len(sys.argv)):
        try:
            frame_index = int(sys.argv[k])
            frame_indices.append(frame_index)
        except:
            print(
                f"{Fore.RED}ERROR: {sys.argv[k]} is not a frame_index in [145942, 351367] U [419700, 624455]{Style.RESET_ALL}"
            )
            print(usage_message)
            sys.exit(1)

    for index in frame_indices:
        subprocess.call(
            [
                "convert",
                "-alpha",
                "on",
                f"{original_dir}/gsw1_{index:06d}.jpg",
                f"{todo_dir}/gsw1_{index:06d}_color.png",
            ]
        )
        subprocess.call(
            [
                "python",
                "black_and_white_mask_to_rgba.py",
                f"{mask_dir}/gsw1_{index:06d}_nonfloor.png",
                f"{todo_dir}/gsw1_{index:06d}_color.png",
                f"{todo_dir}/gsw1_{index:06d}_nonfloor.png",
            ]
        )

    print("On your laptop in iterm, do these commands to download the todo items:")
    print(f"\n    mkdir -p ~/seg/{person}")
    print(f"\n    rsync -r lam:/home/drhea/seg/{person}/ ~/seg/{person}/\n")