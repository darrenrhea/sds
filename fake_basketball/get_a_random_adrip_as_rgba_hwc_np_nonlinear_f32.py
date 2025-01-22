from colorama import Fore, Style
from get_a_random_adrip_file_path import (
     get_a_random_adrip_file_path
)
from prii_linear_f32 import (
     prii_linear_f32
)
from prii import (
     prii
)
from convert_u8_to_linear_f32 import (
     convert_u8_to_linear_f32
)
from augment_texture import (
     augment_texture
)
import numpy as np
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)


def maybe_augment_adrip(
    albu_transform,
    unaugmented_adrip_rgb_hwc_np_u8,
):
    """
    Randomly do or do not augment the texture of the adrip.
    """

    rand_bit = np.random.randint(0, 2)
    
    if rand_bit == 0:
        # print("color of the rip going in as-is, no augmentations.")
        augmented_rgb_hwc_np_u8 = unaugmented_adrip_rgb_hwc_np_u8
    else:
        # print(f"{Fore.GREEN}coin flip tells us to augment, like vary the color or noise of the rip.{Style.RESET_ALL}")
        augmented_rgb_hwc_np_u8 = augment_texture(
            rgb_np_u8=unaugmented_adrip_rgb_hwc_np_u8,
            transform=albu_transform
        )
    
    return augmented_rgb_hwc_np_u8
