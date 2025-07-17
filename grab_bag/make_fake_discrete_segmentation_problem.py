from pathlib import Path
import numpy as np
import PIL.Image
import shutil


def make_fake_discrete_segmentation_problem(desc):
    """
    We need the ability to make a fake dataset for image segmentation such that
    no self-respecting image segmentation technique would fail to see the pattern.
    """
    frame_width = desc["frame_width"]
    frame_height = desc["frame_height"]
    num_training_datapoints = desc["num_training_datapoints"]

    fake_dataset_dir = Path("temp_fake_dataset").resolve()
    shutil.rmtree(fake_dataset_dir, ignore_errors=True)
    fake_dataset_dir.mkdir(exist_ok=True)

    print(f"Sticking an extremely simple, small segmentation problem into {fake_dataset_dir}")
    print("Basically every red pixel should be considered foreground, and every blue pixel should be considered background")

    # Apparently if the dataset is too small, training does not work:
    for i in range(num_training_datapoints):
        mask_bool = np.zeros(
            shape=(frame_height, frame_width),
            dtype=bool
        )

        if i % 2 == 0:
           mask_bool[:(frame_height // 2), :(frame_width // 2)] = 1
        if i % 2 == 1:
           mask_bool[(frame_height // 2):, (frame_width // 2):] = 1


        frame = np.zeros(
            shape=(frame_height, frame_width, 3),
            dtype=np.uint8
        )
        frame[mask_bool] = [255, 0, 0]
        frame[~mask_bool] = [0, 0, 255]

        mask_uint8 = 255 * mask_bool.astype(np.uint8) 
        mask_rgba = np.zeros(
            shape=(frame_height, frame_width, 4),
            dtype=np.uint8
        )
        mask_rgba[..., 3] = mask_uint8
        mask_rgba[..., :3] = frame
        
        frame_path = fake_dataset_dir / f"fake_{i:06d}.jpg"
        mask_path = fake_dataset_dir / f"fake_{i:06d}_nonfloor.png"
        PIL.Image.fromarray(frame).save(frame_path)
        PIL.Image.fromarray(mask_rgba).save(mask_path)

