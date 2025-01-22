from get_augmentation_for_texture import (
     get_augmentation_for_texture
)
from augment_texture import (
     augment_texture
)
from pathlib import Path
import numpy as np
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)



class FloorTextureSource(object):
    """
    This class is for getting a random floor texture
    for Euroleague basketball games, esp. Munich.
    """
    def __init__(self):
        floor_texture_file_paths = []
        floor_textures_dir = Path("~/r/floortextures").expanduser()
        white_list = [
            "23-24_MUNICH_EUR_floortexture.png"
        ]
        for p in floor_textures_dir.glob("*.png"):
            if p.name in white_list:
                floor_texture_file_paths.append(p)

        assert (
            len(floor_texture_file_paths) > 0
        ), f"No floor textures were chosen. Using {white_list} on {floor_textures_dir}"
        self.floor_texture_file_paths = floor_texture_file_paths

    def get_a_random_floor_texture_rgba_np_f32(
        self,
        albu_transform,
        just_solid_black=False
    ):
   
        # choose a random ad from the idiomatic Python list self.ad_names:
        index = np.random.randint(0, len(self.floor_texture_file_paths))
        floor_texture_file_path = self.floor_texture_file_paths[index]
        
        print(f"Chose {floor_texture_file_path}")
        # they never have transparent bits since they are for the LED board:
        unaugmented_texture_rgb_np_u8 = open_as_rgb_hwc_np_u8(floor_texture_file_path)

        texture_rgb_np_u8 = augment_texture(
            rgb_np_u8=unaugmented_texture_rgb_np_u8,
            transform=albu_transform
        )
       
    
        # we have to add an alpha channel to the texture even though it's not transparent:
        texture_rgba_np_u8 = np.zeros(
            shape=(texture_rgb_np_u8.shape[0], texture_rgb_np_u8.shape[1], 4),
            dtype=np.uint8
        )

        texture_rgba_np_u8[:, :, :3] = texture_rgb_np_u8
        texture_rgba_np_u8[:, :, 3] = 255
        
        texture_rgba_np_f32 = texture_rgba_np_u8.astype(np.float32)

        return texture_rgba_np_f32

    
    
if __name__ == "__main__":
    floor_texture_source = FloorTextureSource()
    albu_transform = get_augmentation_for_texture()
    floor_texture_rgba_np_f32 = floor_texture_source.get_a_random_floor_texture_rgba_np_f32(
        albu_transform=albu_transform
    )    
