from PIL import Image
from pathlib import Path
import os
import sys
import numpy as np

# clip_name = "gsw1"
# background_dir = Path(f"~/r/final_gsw1/scorebug_final_segmentation").expanduser()
# foreground_dir = Path(f"~/r/gsw1/scorebug").expanduser()
# out_dir = Path(f"~/r/final_gsw1/scorebug_final_segmentation").expanduser()
background_dir = Path(f"~/thomas_black_completed_46").expanduser()
foreground_dir = Path(f"~/inferred").expanduser()
out_dir = Path(f"~/union").expanduser()
out_dir.mkdir(parents=True, exist_ok=True)

image_ids = [
    # "147568",
    # "150000",
    # "150291",
    # "150331",
    # "150348",
    # "150374",
    # "150893", #
    # "150900", #
    # "150953", #
    # "151004",
    # "151037",
    # "151158",
    # "151359",
    # "151382",
    # "151567",
    # "153700",
    # "153752",
    # "148785",
    # "150000",
    # "150331",
    # "150348",
    # "150374",
    # "150900",
    # "150953",
    # "151004",
    # "151037",
    # "151158",
    # "151359",
    # "151382",
    # "151567",
    # "153700",
    # "153752",
    # "154000",
    # "159064",
    # "159128",
    # "160352",
    # "160800",
    # "161000",
    # "163266",
    # "163576",
    # "163768",
    # "163833",
    # "163851",
    # "163973",
    # "164044",
    # "164244",
    # "164740",
    # "165013",
    # "165719",
    # "165723",
    # "167200",
    # "168178",
    # "169348",
    # "169370",
    # "170501",
    # "170506",
    # "172946",
    # "174565",
    # "193372",
    # "193877",
    # "194127",
    # "194339",
    # "256429",
    # "296895",
    # "303177",
    # "305648",
    # "305706",
    # "328521",
    # "328554",
    # "328980",
    # "331545",
    # "331849",
    # "332931",
    # "333322",
    # "498907",
    # "498934",
    # "499171",
    # "499201",
    # "499227",
    # "499325",
    # "582534",
    # "582538",
    # "585197",
    "585978"
]

# for image_id in image_ids:
for image_file in os.listdir(background_dir):
    print(image_file)
    clip_id = "_".join(str(image_file).split("_")[:-1])
    print(clip_id)
    
    background = Image.open(Path(f"{background_dir}/{clip_id}_nonpad.png").expanduser()).convert('RGBA')
    background_array = np.array(background)
    combined_array = background_array.copy()

    foreground = Image.open(Path(f"{foreground_dir}/{clip_id}_nonfloor.png").expanduser()).convert('RGBA')
    foreground_array = np.array(foreground)

    for i in range(background_array.shape[0]):
        for j in range(background_array.shape[1]):
            combined_array[i, j, 3] = min(background_array[i][j][3], foreground_array[i][j][3])
    
    
    # if clip_id != "liv_BOSvGSW_PGM_core_esp_06-08-2022_262000":
    #     combined_array[965:,:,3] = 255

    combined_array[965:,:,3] = 255
    # Do an alpha composite of foreground over background
    combined_image = Image.fromarray(combined_array)
    combined_image.save(Path(f"{out_dir}/{clip_id}_nonfloor.png"))

    # Image.alpha_composite(foreground, background).save(Path(f"{out_dir}/{clip_id}_nonfloor.png").expanduser())