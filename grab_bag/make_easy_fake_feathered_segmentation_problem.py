from pathlib import Path
import numpy as np
import PIL.Image
import shutil
from prii import prii


def make_easy_fake_feathered_segmentation_problem(
    desc,
    visualize=False
):
    """
    Makes a fake dataset for a feathered image segmentation problem,
    where the pattern is very obvious, easy.

    For both the training data and the inferenced masks
    Every pixel should be assigned a floating point number between 0.0 and 1.0 according to how red it is.
    A pixel which is as red as possible should be assigned a label of 1.0
    albeit the real number 1.0 shall actually be represented as the uint8 value 255
    pixels that do not have any redness should be assigned a label of 0.0, represented by the uint8 0
    and those pixels that have intermediate redness, say 100/255, should be assigned a float label of 100/255, represented as the uint8 100.
    """

    frame_width = desc["frame_width"]
    frame_height = desc["frame_height"]
    num_training_datapoints = desc["num_training_datapoints"]

    fake_dataset_dir = Path("temp_fake_dataset").resolve()
    shutil.rmtree(fake_dataset_dir, ignore_errors=True)
    fake_dataset_dir.mkdir(exist_ok=True)

    print(f"Sticking an extremely simple, small, train-on-feathered segmentation problem into {fake_dataset_dir}")
   
    for i in range(num_training_datapoints):
        mask_u8 = np.zeros(
            shape=(frame_height, frame_width),
            dtype=np.uint8
        )
        i_float, j_float = np.meshgrid(
            np.linspace(-1, 1, frame_width),
            np.linspace(-1, 1, frame_height)
        )

        if np.random.randint(low=0, high=10) < 9:
            angle = np.pi * 2 * i / num_training_datapoints
            a = np.cos(angle) / np.sqrt(2)
            b = np.sin(angle) / np.sqrt(2)
            
            mask_u8 = np.round(
                255 * (
                    1 / (
                        1 + np.exp(
                            - 3 * (a * i_float + b * j_float)
                        )
                    )
                )
            ).astype(np.uint8)
        else:
            mask_u8 = np.random.randint(0, 256, size=(frame_height, frame_width), dtype=np.uint8)
       

        frame = np.zeros(
            shape=(frame_height, frame_width, 3),
            dtype=np.uint8
        )
        # The red channel basically is the label, neural networks should learn just pass it through:
        frame[:, :, 0] = mask_u8

        mask_rgba = np.zeros(
            shape=(frame_height, frame_width, 4),
            dtype=np.uint8
        )
        mask_rgba[..., 3] = mask_u8
        mask_rgba[..., :3] = frame
        
        frame_path = fake_dataset_dir / f"fake_{i:06d}.jpg"
        mask_path = fake_dataset_dir / f"fake_{i:06d}_nonfloor.png"
        PIL.Image.fromarray(frame).save(frame_path)
        PIL.Image.fromarray(mask_rgba).save(mask_path)
        if visualize:
            print(f"Datapoint {i=}")
            prii(frame)
            prii(mask_u8)
       
