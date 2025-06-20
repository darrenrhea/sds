from print_yellow import (
     print_yellow
)
from open_alpha_channel_image_as_a_single_channel_grayscale_image import (
     open_alpha_channel_image_as_a_single_channel_grayscale_image
)
from prii import (
     prii
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from pathlib import Path
from get_datapoint_path_tuples_from_list_of_dataset_folders import (
     get_datapoint_path_tuples_from_list_of_dataset_folders
)
import numpy as np

# mother_data_dir =  Path("/shared").expanduser()
mother_data_dir =  Path("~/r").expanduser()

if True:
    specialization_court_dataset_folders = [
        mother_data_dir / "ind-lal-2023-02-02-mxf_floor/.approved",
        mother_data_dir / "ind-bos-2024-10-30-hack_floor/.approved",
        mother_data_dir / "ind-okc-2025-06-11-hack_floor/.approved",
        mother_data_dir / "ind-okc-2025-06-11-hack_2_floor/.approved",
        mother_data_dir / "ind-okc-2025-06-11-hack_3_floor/.approved",
        mother_data_dir / "okc-phi-2022-12-31-mxf_floor/.approved",
    ]


    for d in specialization_court_dataset_folders:
        assert d.is_dir(), f"Every dataset_folder must be a directory, but {d=} is not a directory"


    specialization_court_datapoint_path_tuples = get_datapoint_path_tuples_from_list_of_dataset_folders(
        dataset_folders=specialization_court_dataset_folders,
    )

    print_yellow(
        f"specialization_court_datapoint_path_tuples: found {len(specialization_court_datapoint_path_tuples)} datapoint path tuples from {len(specialization_court_dataset_folders)} dataset folders"
    )
    datapoint_path_tuples = specialization_court_datapoint_path_tuples

else:
    frame_indices = [3779, ]
    final_model_id = "pacersrev7epoch16"
    clip_id = "ind-okc-2025-06-11-hack"

    preann_dir = Path(
        f"~/a/preannotations/fixups/{clip_id}/{final_model_id}"
    ).expanduser()
    datapoint_path_tuples = []
    for frame_index in frame_indices:
        original_path = preann_dir / f"{clip_id}_{frame_index:06d}_original.jpg"
        mask_path = preann_dir / f"{clip_id}_{frame_index:06d}_nonfloor.png"
        assert original_path.is_file(), f"ERROR: {original_path} is not a file!"
        assert mask_path.is_file(), f"ERROR: {mask_path} is not a file!"
        datapoint_path_tuples.append(
            (original_path, mask_path, None)  # no relevance mask
        )

for original_path, mask_path, _ in datapoint_path_tuples:
    print_yellow(mask_path)
    rgb_hwc_np_u8 = open_as_rgb_hwc_np_u8(
        image_path=original_path
    )
    mask_hw_np_u8 = open_alpha_channel_image_as_a_single_channel_grayscale_image(
        abs_file_path=mask_path
    )
    rgba_hwc_np_u8 = np.zeros(
        shape=(
            rgb_hwc_np_u8.shape[0],
            rgb_hwc_np_u8.shape[1],
            4
        ),
        dtype=np.uint8
    )
    rgba_hwc_np_u8[:, :, :3] = rgb_hwc_np_u8
    rgba_hwc_np_u8[:, :, 3] = mask_hw_np_u8

    prii(rgb_hwc_np_u8, caption="original")
    # prii(mask_hw_np_u8, caption="mask_hw_np_u8")

    rgb_hwc_np_f32 = rgb_hwc_np_u8.astype(np.float32)
    blue = rgb_hwc_np_f32[:, :, 2]

    # radius = 3
    # centroid = np.array(
    #     [
    #         51, 58, 53
    #     ],
    #     dtype=np.float32
    # )

    # diff = rgb_hwc_np_f32 - centroid
    threshold = 49
    radius = np.sqrt(threshold**2 + threshold**2 + threshold**2)
    squared = rgb_hwc_np_f32 ** 2
    squared_sum = np.sum(squared, axis=-1)
    indicator = np.logical_and(
        squared_sum < radius ** 2,
        blue < 40
    )
    change_indicator = np.logical_and(
        indicator,
        mask_hw_np_u8 < 64
    )
    # change_indicator = changed.astype(np.uint8) * 255
    # prii(change_indicator)  # We are about to see it in bright green, that is enough
    rgb_hwc_np_u8[change_indicator, :] = [0, 255, 0]
    prii(rgb_hwc_np_u8, caption="Where is would be changed")

    rgba_hwc_np_u8[change_indicator, 3] = 255
    # prii(rgba_hwc_np_u8, caption="rgba_hwc_np_u8")