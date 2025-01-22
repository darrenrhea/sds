from pathlib import Path
import PIL
import PIL.Image
import numpy as np
from print_image_in_iterm2 import print_image_in_iterm2

print_images = False

inferenced_directory = Path(
    "~/wood_light_normal_temp"
).expanduser()

annotation_ids = []
for p in inferenced_directory.iterdir():
    if not p.is_file():
        continue
    if not p.name.endswith("_nonwood.png"):
        continue

    annotation_id = p.stem[:-8]
    print(annotation_id)
    annotation_ids.append(annotation_id)
    

model_ids = [
    "17c5-0307-175f-a850-no-overlap",
    "fe73-e72c-500f-f8bb-no-overlap",
]



for annotation_id in annotation_ids:        

    gold_standard_path = inferenced_directory / f"{annotation_id}_nonwood.png"


    gold_standard_pil = PIL.Image.open(gold_standard_path)
    gold_standard_np = (np.array(gold_standard_pil)[:,:,-1] > 127).astype(np.uint8)

    if print_images:
        print("This is the gold standard from human annnotators:")
        print_image_in_iterm2(grayscale_np_uint8=gold_standard_np * 255)

    for model_id in model_ids:
        inferred_path = inferenced_directory / f"{annotation_id}_{model_id}.png"

        inferred_pil = PIL.Image.open(inferred_path)
        inferred_np = (np.array(inferred_pil)[:,:,-1] > 127).astype(np.uint8)

        if print_images:
            print(f"This is what the model {model_id} says")
            print_image_in_iterm2(grayscale_np_uint8=inferred_np*255)

        diff_np = np.zeros(
            shape=(gold_standard_pil.height, gold_standard_pil.width, 3),
            dtype=np.uint8
        )
        where_red = gold_standard_np < inferred_np 
        where_blue = gold_standard_np > inferred_np
        num_red = np.sum(where_red)
        num_blue = np.sum(where_blue)
        diff_np[where_red,:] = [255, 0, 0] 
        diff_np[where_blue,:] = [0, 0, 255] 

        print(f"Blue means the model left out part of the foreground object. Red means it added too much to the foreground, like a halo")

        print_image_in_iterm2(rgb_np_uint8=diff_np)
        print(f"{annotation_id=} {num_blue=}, {num_red=}")