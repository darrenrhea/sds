import sys
from write_rgb_np_u8_to_png import (
     write_rgb_np_u8_to_png
)
from pathlib import Path
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
import numpy as np
import PIL.Image
        
import PIL.ImageFilter

def add_random_color_spill(
    image_path: Path,
    out_path: Path
):

    block_size = np.random.randint(10, 20)

    h = (1080 + block_size - 1) // block_size
    w = (1920 + block_size - 1) // block_size

    r = np.zeros(
        shape=(h, w),
        dtype=np.uint8
    )
    frequency = 0.2 + np.random.rand() * 0.4

    r[:,:] = np.random.rand(h, w) < frequency

    b = r.repeat(block_size, axis=0).repeat(block_size, axis=1)[:1080, :1920]
    
    random_pil = PIL.Image.fromarray(b*255)

    blur_filter = PIL.ImageFilter.GaussianBlur(
        radius=block_size//2
    )
        
    augmented_pil = random_pil.filter(filter=blur_filter)

    pattern_np = np.array(augmented_pil)

    # green = pattern_np[:,:, np.newaxis].repeat(3, axis=2)
    # green[:,:,0] = 0
    # green[:,:,2] = 0
    # prii(green // 16 )
    rgb = open_as_rgb_hwc_np_u8(image_path)
    result_i16 = rgb.astype(np.int16)
    subtract = np.random.randint(0, 16)
    result_i16[:,:,1] += pattern_np.astype(np.int16) // 16 - subtract
    result_u8 = np.clip(result_i16, 0, 255).astype(np.uint8)
    # prii(
    #     result_u8
    # )
    write_rgb_np_u8_to_png(
        rgb_hwc_np_u8=result_u8,
        out_abs_file_path=out_path,
        verbose=False
    )
    print(f"Wrote {out_path}")





def demo_yo():
    image_path = Path(
        "~/cs/bos-mia-2024-04-21-mxf_440565_original.jpg"
    ).expanduser()

    out_path = Path(
        "~/cs/bos-mia-2024-04-21-mxf_440565_z.jpg"
    ).expanduser()
    
    add_random_color_spill(
        image_path=image_path,
        out_path=out_path
    )

def process_all(
    image_dir: Path,
    out_dir: Path
):
    assert image_dir.is_dir(), f"{image_dir} is not a directory"
    assert out_dir.is_dir(), f"{out_dir} is not a directory"
    image_paths = [
        x for x in
        image_dir.glob("*_original.png")
    ]
    for index, image_path in enumerate(image_paths):
        print(f"Processing {index+1}/{len(image_paths)}: {image_path}")
        out_path = out_dir / image_path.name
        add_random_color_spill(
            image_path=image_path,
            out_path=out_path
        )


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <image_dir> <out_dir>")
        print("Example: python add_random_color_spill.py /shared/fake_nba/ESPN_DAL_LAC_NEXT_ABC /shared/fake_nba/ESPN_DAL_LAC_NEXT_ABC_green")
        sys.exit(1)
    image_dir = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    process_all(
        image_dir=image_dir,
        out_dir=out_dir
    )