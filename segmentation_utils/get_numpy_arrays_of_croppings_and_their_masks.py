import time
import numpy as np
from cut_this_many_interesting_subrectangles_from_annotated_image import cut_this_many_interesting_subrectangles_from_annotated_image
from cut_this_many_interesting_subrectangles_from_annotated_image import cut_this_many_interesting_subrectangles_from_annotated_image_bw


def get_numpy_arrays_of_croppings_and_their_masks(
    list_of_annotated_images,
    crop_height,
    crop_width,
    desired_mask_names,
    mask_name_to_min_amount,
    mask_name_to_max_amount,
    how_many_originals,
    how_many_crops_per_original,
    mask_encoding_type
):
    """
    The reason that the arguments are almost the same as
    cut_this_many_interesting_subrectangles_from_annotated_image is that
    this calls cut_this_many_interesting_subrectangles_from_annotated_image
    once for each annotated_image, then stacks the resulting croppings and
    cropped masks into numpy arrays.

    We want the training data to be stacked into, first, a np.uint8 numpy array
    cropped_originals
    of
    tens of thousands of rgb images, and second,
    some another np.uint8 np.arrays
    of cropped_masks of various sizes
    of the corresponding binary segmentation masks.
    This can take a long time, for 20 crops from 1600 images it takes 30 minutes.
    Note making crops is relatively cheap compared to the loading of an image,
    so how_many_crops_per_original=40 does not take 10 times longer than how_many_crops_per_original=4
    """
    assert (
        isinstance(list_of_annotated_images, list)
    ), "list_of_annotated_images must be a list"

    assert (
        how_many_originals <= len(list_of_annotated_images)
    ), f"how_many_originals must be less than len(list_of_annotated_images) = {len(list_of_annotated_images)}"

    start_time = time.time()  # start timing
    num_crops = how_many_originals * how_many_crops_per_original

    # preallocate for color croppings.  We may cut excess allocation off towards the end of this procedure:
    cropped_originals = np.zeros(
        shape=(num_crops, crop_height, crop_width, 3),
        dtype=np.uint8
    )

    # preallocate for croppings of each type of mask.  We may cut excess allocation off towards the end of this procedure:
    mask_name_to_cropped_masks = dict()
    for mask_name in desired_mask_names:
        mask_name_to_cropped_masks[mask_name] = np.zeros(
            shape=(num_crops, crop_height, crop_width),
            dtype=np.uint8
        )

    how_many_already_cut = 0  # since success is not guaranteed, we count how many croppings we have pulled off successfully so far.
    for image_index in range(how_many_originals):
        if image_index % 1 == 0:
            print(f"{image_index} originals finished in {(time.time() - start_time)/60} minutes")

        dct = cut_this_many_interesting_subrectangles_from_annotated_image(
            annotated_image=list_of_annotated_images[image_index],
            how_many_croppings_to_cut_out=how_many_crops_per_original,
            crop_height=crop_height,
            crop_width=crop_width,
            desired_mask_names=desired_mask_names,
            mask_name_to_min_amount=mask_name_to_min_amount,
            mask_name_to_max_amount=mask_name_to_max_amount,
            mask_encoding_type=mask_encoding_type
        )

        if not dct["success"]:
            continue  # we couldn't manage to cut any croppings from this image satisfying the requirements
        how_many_cut_from_this_image = dct["num_croppings"]

        # stick the color croppings into the numpy array
        cropped_originals[
            how_many_already_cut: how_many_already_cut + how_many_cut_from_this_image,
            :,
            :,
            :
        ] = dct["cropped_originals"]
        # print(f"cropped originals {cropped_originals}")

        # stick the croppings of various types of masks into their respective numpy arrays:
        for mask_name in desired_mask_names:
            mask_name_to_cropped_masks[mask_name][
                how_many_already_cut: how_many_already_cut + how_many_cut_from_this_image,
                :,
                :
            ] = dct["mask_name_to_cropped_masks"][mask_name]
        how_many_already_cut += how_many_cut_from_this_image
    
    num_croppings_cut = how_many_already_cut

    # we may have over-allocated.
    # given that it is possible to fail to cut as many cropping as we wanted, we need to remove the excess allocation:
    cropped_originals = cropped_originals[:num_croppings_cut, :, :, :].copy()
    for mask_name in desired_mask_names:
        mask_name_to_cropped_masks[mask_name] = mask_name_to_cropped_masks[mask_name][
            :num_croppings_cut,
            :,
            :
        ].copy()
    
    stop_time = time.time()
    
    print(
        f"Took {(stop_time - start_time)/ 60} minutes to take {how_many_crops_per_original} from each of {how_many_originals} images"
    )
    
    return dict(
        num_croppings_cut=num_croppings_cut,
        mask_name_to_cropped_masks=mask_name_to_cropped_masks,
        cropped_originals=cropped_originals
    )


def get_numpy_arrays_of_croppings_and_their_masks_bw(
    list_of_annotated_images,
    crop_height,
    crop_width,
    desired_mask_names,
    mask_name_to_min_amount,
    mask_name_to_max_amount,
    how_many_originals,
    how_many_crops_per_original,
    mask_encoding_type
):
    """
    The reason that the arguments are almost the same as
    cut_this_many_interesting_subrectangles_from_annotated_image is that
    this calls cut_this_many_interesting_subrectangles_from_annotated_image
    once for each annotated_image, then stacks the resulting croppings and
    cropped masks into numpy arrays.

    We want the training data to be stacked into, first, a np.uint8 numpy array
    cropped_originals
    of
    tens of thousands of rgb images, and second,
    some another np.uint8 np.arrays
    of cropped_masks of various sizes
    of the corresponding binary segmentation masks.
    This can take a long time, for 20 crops from 1600 images it takes 30 minutes.
    Note making crops is relatively cheap compared to the loading of an image,
    so how_many_crops_per_original=40 does not take 10 times longer than how_many_crops_per_original=4
    """
    assert (
        isinstance(list_of_annotated_images, list)
    ), "list_of_annotated_images must be a list"

    assert (
        how_many_originals <= len(list_of_annotated_images)
    ), f"how_many_originals must be less than len(list_of_annotated_images) = {len(list_of_annotated_images)}"

    start_time = time.time()  # start timing
    num_crops = how_many_originals * how_many_crops_per_original

    # preallocate for black and white croppings.  We may cut excess allocation off towards the end of this procedure:
    cropped_originals = np.zeros(
        shape=(num_crops, crop_height, crop_width),
        dtype=np.uint8
    )

    # preallocate for croppings of each type of mask.  We may cut excess allocation off towards the end of this procedure:
    mask_name_to_cropped_masks = dict()
    for mask_name in desired_mask_names:
        mask_name_to_cropped_masks[mask_name] = np.zeros(
            shape=(num_crops, crop_height, crop_width),
            dtype=np.uint8
        )

    how_many_already_cut = 0  # since success is not guaranteed, we count how many croppings we have pulled off successfully so far.
    for image_index in range(how_many_originals):
        if image_index % 1 == 0:
            print(f"{image_index} originals finished in {(time.time() - start_time)/60} minutes")

        dct = cut_this_many_interesting_subrectangles_from_annotated_image_bw(
            annotated_image=list_of_annotated_images[image_index],
            how_many_croppings_to_cut_out=how_many_crops_per_original,
            crop_height=crop_height,
            crop_width=crop_width,
            desired_mask_names=desired_mask_names,
            mask_name_to_min_amount=mask_name_to_min_amount,
            mask_name_to_max_amount=mask_name_to_max_amount,
            mask_encoding_type=mask_encoding_type
        )

        if not dct["success"]:
            continue  # we couldn't manage to cut any croppings from this image satisfying the requirements
        how_many_cut_from_this_image = dct["num_croppings"]

        # stick the color croppings into the numpy array
        cropped_originals[
            how_many_already_cut: how_many_already_cut + how_many_cut_from_this_image,
            :,
            :
        ] = dct["cropped_originals"]
        # print(f"cropped originals {cropped_originals}")

        # stick the croppings of various types of masks into their respective numpy arrays:
        for mask_name in desired_mask_names:
            mask_name_to_cropped_masks[mask_name][
                how_many_already_cut: how_many_already_cut + how_many_cut_from_this_image,
                :,
                :
            ] = dct["mask_name_to_cropped_masks"][mask_name]
        how_many_already_cut += how_many_cut_from_this_image
    
    num_croppings_cut = how_many_already_cut

    # we may have over-allocated.
    # given that it is possible to fail to cut as many cropping as we wanted, we need to remove the excess allocation:
    cropped_originals = cropped_originals[:num_croppings_cut, :, :].copy()
    for mask_name in desired_mask_names:
        mask_name_to_cropped_masks[mask_name] = mask_name_to_cropped_masks[mask_name][
            :num_croppings_cut,
            :,
            :
        ].copy()
    
    stop_time = time.time()
    
    print(
        f"Took {(stop_time - start_time)/ 60} minutes to take {how_many_crops_per_original} from each of {how_many_originals} images"
    )

    print(f"cropped originals shape {cropped_originals.shape}")
    
    return dict(
        num_croppings_cut=num_croppings_cut,
        mask_name_to_cropped_masks=mask_name_to_cropped_masks,
        cropped_originals=cropped_originals
    )
