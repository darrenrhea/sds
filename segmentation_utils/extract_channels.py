import sys
import PIL
import PIL.Image
import numpy as np
from pathlib import Path
from numpy.core.fromnumeric import searchsorted
from numpy.lib.npyio import save


def extract_channels(image_path):

    size_counts = list(range(256))

    image_pil = PIL.Image.open(str(image_path))
    image_np = np.array(image_pil)

    sizes_dict = dict()

    # for i in range(image_np.shape[0]):
    #     for j in range(image_np.shape[1]):
    #         if image_np[i, j] == 127:
    #             print(f"{i} {j} pixel is 127")
    
    for size in [image_np[i, j] for i in range(image_np.shape[0]) for j in range(image_np.shape[1])]:
        if size not in sizes_dict.keys():
            sizes_dict[size] = 1
        else:
            sizes_dict[size] += 1

    sorted_sizes_dict = dict(sorted(sizes_dict.items()))
    print(f"sizes dictionary {sorted_sizes_dict}")

    
    # save_pil = PIL.Image.fromarray(rgba_np)
    # save_pil.save(save_path)


if __name__ == "__main__":
   
    image_path = Path(sys.argv[1])
    
    extract_channels(
        image_path=image_path
    )
