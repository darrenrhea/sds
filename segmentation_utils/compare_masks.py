from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio
import better_json
import sys

json_config_path = sys.argv[1]
json_config_file = Path(json_config_path).expanduser()
config_dict = better_json.load(json_config_file)
ground_truth_image_path = Path(f"{config_dict['ground_truth_image_path']}").expanduser()
inferred_image_path = Path(f"{config_dict['inferred_image_path']}").expanduser()
out_prefix = config_dict['out_prefix']
image_out_path = config_dict['image_out_path']
original_image_path = config_dict['original_image_path']
gif_out_path = config_dict['gif_out_path']

# ground_truth_image_path = Path(f"~/r/brooklyn_nets_barclays_center/nonfloor_segmentation_test").expanduser()
# deeplabv3_inferred_image_path = Path(f"~/r/segmentation_utils/deeplabv3_temp").expanduser()
name_to_array = {}
for test_file in ground_truth_image_path.iterdir():
    if str(test_file).endswith("nonfloor.png"):
        test_file_name = str(test_file).split("/")[-1].rsplit('_', 1)[0]
        print(f"test file name {test_file_name}")
        name_to_array[test_file_name] = {}
        test_pil = Image.open(test_file).convert('RGBA')
        test_array = np.array(test_pil)
        name_to_array[test_file_name]["test_result_array"] = (test_array[:, :, 3] > 128).astype(np.uint8)

for inferred_file in inferred_image_path.iterdir():
    inferred_file_name = inferred_file.stem
    inferred_pil = Image.open(inferred_file).convert('L')
    inferred_array = np.array(inferred_pil)
    name_to_array[inferred_file_name]["inferred_array"] = (inferred_array[:, :] > 128).astype(np.uint8)

tot_num_pixels = 1080*1920
pixel_comparisons = {}
for name in name_to_array.keys():
    pixel_comparisons[name] = {}
    pixel_comparisons[name]["test_vs_inferred"] = {}
    pixel_comparisons[name]["test_vs_inferred"]["same"] = (name_to_array[name]["test_result_array"] == name_to_array[name]["inferred_array"]).sum()
    pixel_comparisons[name]["test_vs_inferred"]["different"] = (name_to_array[name]["test_result_array"] != name_to_array[name]["inferred_array"]).sum()

for name in pixel_comparisons.keys():
    num_same = pixel_comparisons[name]['test_vs_inferred']['same']
    num_different = pixel_comparisons[name]['test_vs_inferred']['different']
    assert num_different == tot_num_pixels - num_same
    print(f"{name}:")
    print(f" same {num_same}")
    print(f" different {num_different}")
    print(f" PERCENT CORRECT {round((num_same/tot_num_pixels)*100, 2)}")

num_rows = 1080
num_columns = 1920
compare_array = np.zeros(shape=(num_rows, num_columns, 3), dtype=np.uint8)
foreground_color = config_dict['foreground_color']
background_color = config_dict['background_color']
foreground_as_background_color = config_dict['foreground_as_background_color']
background_as_foreground_color = config_dict['background_as_foreground_color']
# foreground_color = [255, 255, 255]
# background_color = [0, 0, 0]
# foreground_as_background_color = [0, 255, 0]
# background_as_foreground_color = [255, 0, 0]
for name in pixel_comparisons.keys():
    filenames = []
    inferred_fgbg_array = name_to_array[name]["inferred_array"]
    test_fgbg_array = name_to_array[name]["test_result_array"]
    # red is background labeled as foreground
    # green is foreground labeled as background
    category_array = 2*inferred_fgbg_array + test_fgbg_array
    background_indices = (category_array == 0)
    foreground_as_background_indices = (category_array == 1)
    background_as_foreground_indices = (category_array == 2)
    foreground_indices = (category_array == 3)
    compare_array[foreground_indices, :] = foreground_color[:]
    compare_array[background_indices, :] = background_color[:]
    compare_array[foreground_as_background_indices, :] = foreground_as_background_color[:]
    compare_array[background_as_foreground_indices, :] = background_as_foreground_color[:]
    print(f"saving image {name}")
    compare_image = Image.fromarray(compare_array, 'RGB')
    out_image = Path(f"{image_out_path}/{out_prefix}_{name}.png").expanduser()
    original_image = Path(f"{original_image_path}/{name}_color.png").expanduser()
    # image_path = Path(f"~/r/segmentation_utils/deeplabv3_vs_test_comparisons/deeplabv3_vs_test_{name}.png").expanduser()
    # original_image_path = Path(f"~/r/brooklyn_nets_barclays_center/nonfloor_segmentation_test/{name}_color.png").expanduser()
    compare_image.save(out_image)
    filenames.append(out_image)
    filenames.append(original_image)
    with imageio.get_writer(Path(f"{gif_out_path}/{out_prefix}_{name}.gif"), mode='I', duration=1) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
