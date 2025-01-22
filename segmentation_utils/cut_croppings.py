from annotated_data import get_list_of_annotated_images_from_several_directories
from all_imports_for_image_segmentation import *
from pathlib import Path

crop_height = 400
crop_width = 400
regenerate_crops = True
desired_mask_names = ["nonfloor"]
target_mask = "nonfloor"

# if downsampling, make sure to downsample the original training masks and point the function
# get_list_of_annotated_images() to the directory containing the downsampled masks.
list_of_annotated_images = get_list_of_annotated_images_from_several_directories(
    must_have_these_masks=desired_mask_names,
    directories_to_gather_from_with_limits = [
        # (
        #     Path(f"~/r/final_gsw1/GSWvBOS_06-05-2022_PGM_ESP_MXF").expanduser(),
        #     1
        # ),
        # (
        #     Path(f"~/r/final_gsw1/nonfloor_convention").expanduser(),
        #     1
        # ),
        (
            Path(f"~/r/brooklyn_nets_barclays_center/nonfloor_segmentation_train").expanduser(),
            1000000
        )
    ]
)

num_frames = len(list_of_annotated_images)

pp.pprint(list_of_annotated_images)

print(f"num_frames = {num_frames}")

# croppings_dir = "~/gsw/400x400"
croppings_dir = "~/brooklyn_nets_barclays_center/fastai_vs_deeplabv3_400x400"
Path(croppings_dir).expanduser().mkdir(parents=True, exist_ok=True)
start = time.time()
if regenerate_crops:
    cropped_hand_annotated_training_data = get_numpy_arrays_of_croppings_and_their_masks(
        list_of_annotated_images=list_of_annotated_images,
        crop_height=crop_height,
        crop_width=crop_width,
        desired_mask_names=desired_mask_names,
        mask_name_to_min_amount=dict(nonfloor=1000),
        mask_name_to_max_amount=dict(nonfloor=crop_height*crop_width - 1),
        how_many_originals=len(list_of_annotated_images),
        how_many_crops_per_original=1000,
        mask_encoding_type="alpha_channel_of_rgba"
    )

    dct = cropped_hand_annotated_training_data
    print([k for k in dct.keys()])
    assert dct["num_croppings_cut"] > 0
    assert isinstance(dct["mask_name_to_cropped_masks"], dict)
    assert isinstance(dct["cropped_originals"], np.ndarray)
    for mask_name in desired_mask_names:
    #     print(f"We have the masks known as {mask_name}")
        assert dct["mask_name_to_cropped_masks"][mask_name].shape[0] == dct["cropped_originals"].shape[0]
    for k in range(dct["num_croppings_cut"]):
        
        save_hwc_np_uint8_to_image_path(dct["cropped_originals"][k], Path(f"{croppings_dir}/{k}_color.png").expanduser())

        save_hwc_np_uint8_to_image_path(
            dct["mask_name_to_cropped_masks"][target_mask][k],
            Path(f"{croppings_dir}/{k}_{target_mask}.png").expanduser())

        if (k%1000 == 0):
            print(f"saving {k}")
            update = time.time()
            print(f"{update - start} seconds")
    end = time.time()
    print(f"total {end - start} seconds")