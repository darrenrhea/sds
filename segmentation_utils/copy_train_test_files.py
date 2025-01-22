import os
from pathlib import Path
import shutil


shared_path = Path(f"~/r/brooklyn_nets_barclays_center/").expanduser()
all_name = Path(f"nonfloor_segmentation").expanduser()
test_name = Path(f"nonfloor_segmentation_test").expanduser()
train_name = Path(f"nonfloor_segmentation_train").expanduser()
original_path = shared_path / all_name
test_path = shared_path / test_name
train_path = shared_path / train_name
file_counter = 0
for my_file in os.listdir(original_path):
    if my_file.endswith("nonfloor.png"):
        file_id = my_file.rsplit('_', 1)[0]
        nonfloor_name = Path(file_id + "_nonfloor.png").expanduser()
        color_name = Path(file_id + "_color.png").expanduser()
        inbounds_name = Path(file_id + "_inbounds.png").expanduser()
        if file_counter % 10 == 0:
            shutil.copy(original_path / nonfloor_name, test_path / nonfloor_name)
            shutil.copy(original_path / color_name, test_path / color_name)
            shutil.copy(original_path / inbounds_name, test_path / inbounds_name)
        else:     
            shutil.copy(original_path / nonfloor_name, train_path / nonfloor_name)
            shutil.copy(original_path / color_name, train_path / color_name)
            shutil.copy(original_path / inbounds_name, train_path / inbounds_name)
        file_counter += 1