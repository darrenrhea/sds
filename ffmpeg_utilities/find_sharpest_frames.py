import subprocess
from pathlib import Path
import numpy as np
import sys
import cv2
import PIL.Image

def get_sharpness(image_np) -> float:
    grayscale = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    assert grayscale.shape == (1080, 1920), f"ERROR: {grayscale.shape}"
    return cv2.Laplacian(grayscale, cv2.CV_64F).var()

def find_sharpest_frames_cli():
   
    frames_dir = Path(
        "temp"
    ).resolve()
   
    clip_id = ""

    for frame_index in range(0, 1000_0000):
        print(f"frame_index: {frame_index}")
        frame_path = frames_dir / f"{clip_id}_{frame_index:06d}.jpg"
        if not frame_path.exists():
            break
        image_pil = PIL.Image.open(frame_path)
        image_np = np.array(image_pil)
        sharpness = get_sharpness(image_np)
        print(f"sharpness: {sharpness}")
    
if __name__ == "__main__":
    find_sharpest_frames_cli()
