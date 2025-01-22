import numpy as np
import PIL
import PIL.Image
from image_loaders_and_savers import (
    get_hwc_np_uint8_from_image_path,
    get_hwc_np_uint8_from_image_path_bw
)


def cut_this_many_interesting_subrectangles_from_annotated_image(
    annotated_image,
    how_many_croppings_to_cut_out,
    crop_height,
    crop_width,
    desired_mask_names,
    mask_name_to_min_amount,
    mask_name_to_max_amount,
    mask_encoding_type,
):
    """
    Loading a single 4K or FHD (3840x2160 or 1920x1080) image from disk is a bit slow,
    so we want to create several (approx one hundred) croppings from it all at once, say 256x256 croppings.
    You might also want to ensure that the croppings have a large amount of certain objects in them,
    such as grass, adwall, foreground, or maybe you might want to ensure not too much
    of a certain segment.  For instance, 100% on grass is kinda boring and is almost a chroma-keyable issue.

    Returns a dictionary contained the following items:
    cropped_originals, a uint8 numpy array of shape=
    (how_many_croppings_to_cut_out, crop_height, crop_width, 3 channels)

    and mask_name_to_cropped_masks, a dictionary from each key in desired_mask_names to
    a uint8 numpy array of shape =
    (how_many_croppings_to_cut_out, crop_height, crop_width) np.uint8 image
    which is an indicator of whether each pixel is in that particular mask.
    (values are only 0 or 1, 1 means in the mask, like is grass, is adwall, etc.)

    """
    assert (
        set(mask_name_to_min_amount.keys()).issubset(desired_mask_names)
    ), "mask_name_to_min_amount is making a requirement on a mask you did not request.  Not allowed"

    assert (
        set(mask_name_to_max_amount.keys()).issubset(desired_mask_names)
    ), "mask_name_to_max_amount is making a requirement on a mask you did not request.  Not allowed"

    possible_mask_encoding_types = ["alpha_channel_of_rgba", "mask_is_anything_not_pure_black"]
    assert (
        mask_encoding_type in possible_mask_encoding_types
    ), f"mask_encoding_type must be amongst {possible_mask_encoding_types}, not {mask_encoding_type}"

    possible_mask_names = [
        "nonpad",
        "nonwood",
        "nonfloor",
        "nonled",
        "mainrectangle",
        "adwall",
        "foreground",
        "grass",
        "relevant",
        "nonlane",
        "relevant_lane",
        "avg_rhs_lane_mask",
        "players",
        "inbounds"
    ]
    max_iters = 100000

    mask_name_to_mask_path = annotated_image["mask_name_to_mask_path"]

    assert isinstance(mask_name_to_min_amount, dict)
    assert set(mask_name_to_min_amount.keys()).issubset(possible_mask_names)

    assert isinstance(mask_name_to_max_amount, dict)
    assert set(mask_name_to_max_amount.keys()).issubset(possible_mask_names)

    assert set(desired_mask_names).issubset(possible_mask_names)

    original_hwc_uint8_np = get_hwc_np_uint8_from_image_path(
        image_path=annotated_image['image_path'])
    height, width, _ = original_hwc_uint8_np.shape

    mask_name_to_fullsized_mask = dict()
    for mask_name in desired_mask_names:
        mask_name_to_fullsized_mask[mask_name] = np.zeros(shape=(height, width), dtype=np.uint8)

        mask_path = mask_name_to_mask_path[mask_name]
        # we need to have Blender do the emission trick to make this reliable, the black-pentagoned soccer ball does not have holes in it!
        if mask_encoding_type == "mask_is_anything_not_pure_black":
            segmentation_hwc_uint8_np = get_hwc_np_uint8_from_image_path(
                image_path=mask_path
            )

            mask_name_to_fullsized_mask[mask_name][:, :] = np.max(segmentation_hwc_uint8_np, axis=2) >= 2

        elif mask_encoding_type == "alpha_channel_of_rgba":
            segmentation_hwc_uint8_np = np.array(
                PIL.Image.open(str(mask_path))
            )  # we wish that this only took values 0 or 255
            print(f'mask path is {mask_path}')
            mask_name_to_fullsized_mask[mask_name][:, :] = (segmentation_hwc_uint8_np[:, :, 3] > 128)

    cropped_originals = np.zeros(shape=(how_many_croppings_to_cut_out, crop_height, crop_width, 3), dtype=np.uint8)
    mask_name_to_cropped_masks = dict()
    for mask_name in desired_mask_names:
        mask_name_to_cropped_masks[mask_name] = np.zeros(shape=(how_many_croppings_to_cut_out, crop_height, crop_width), dtype=np.uint8)

    for mask_name in desired_mask_names:
        num_pixels_on_in_mask = np.sum(mask_name_to_fullsized_mask[mask_name])
        # print(f"num pixels on on mask {num_pixels_on_in_mask}")
        assert (
            num_pixels_on_in_mask > 400
        ), f"image {annotated_image['image_path']} has almost no {mask_name}: only {num_pixels_on_in_mask} pixels are on in it."

    possible = True  # sometimes you ask for something that cannot be satisfied or is unlikely to be satisfied within maxiters

    for cntr in range(how_many_croppings_to_cut_out):
        iters = 0
        while True:
            # choose a cropping at random
            i0 = np.random.randint(0, height - crop_height + 1)
            j0 = np.random.randint(0, width - crop_width + 1)

            # determine if it is acceptable, i.e. if we have at least the minimal amount for all masks
            acceptable = True
            for mask_name in desired_mask_names:  
                s = np.sum(mask_name_to_fullsized_mask[mask_name][i0:i0 + crop_height, j0:j0 + crop_width])
                if (
                    s < mask_name_to_min_amount.get(mask_name, 0)
                    or
                    s > mask_name_to_max_amount.get(mask_name, crop_height * crop_width)
                ):
                    acceptable = False
                    break

            if acceptable:  # found a good i0, j0 that specifies a cropping meeting the requirements
                break

            iters += 1
            if iters > max_iters:
                possible = False
                break

        if not possible:
            return dict(
                success=False,
                num_croppings=0,
                message=f"You might check out image {annotated_image}, it doesnt seem easy to find croppings satisfying your requirements"
            )

        cropped_originals[cntr, :, :, :] = \
            original_hwc_uint8_np[i0: i0 + crop_height, j0: j0 + crop_width, :]

        for mask_name in desired_mask_names:
            mask_name_to_cropped_masks[mask_name][cntr, :, :] = \
                mask_name_to_fullsized_mask[mask_name][i0: i0 + crop_height, j0: j0 + crop_width]

    return dict(
        success=True,
        num_croppings=how_many_croppings_to_cut_out,
        cropped_originals=cropped_originals,
        mask_name_to_cropped_masks=mask_name_to_cropped_masks
    )

def cut_this_many_interesting_subrectangles_from_annotated_image_bw(
    annotated_image,
    how_many_croppings_to_cut_out,
    crop_height,
    crop_width,
    desired_mask_names,
    mask_name_to_min_amount,
    mask_name_to_max_amount,
    mask_encoding_type,
):
    """
    Loading a single 4K or FHD (3840x2160 or 1920x1080) image from disk is a bit slow,
    so we want to create several (approx one hundred) croppings from it all at once, say 256x256 croppings.
    You might also want to ensure that the croppings have a large amount of certain objects in them,
    such as grass, adwall, foreground, or maybe you might want to ensure not too much
    of a certain segment.  For instance, 100% on grass is kinda boring and is almost a chroma-keyable issue.

    Returns a dictionary contained the following items:
    cropped_originals, a uint8 numpy array of shape=
    (how_many_croppings_to_cut_out, crop_height, crop_width)

    and mask_name_to_cropped_masks, a dictionary from each key in desired_mask_names to
    a uint8 numpy array of shape =
    (how_many_croppings_to_cut_out, crop_height, crop_width) np.uint8 image
    which is an indicator of whether each pixel is in that particular mask.
    (values are only 0 or 1, 1 means in the mask, like is grass, is adwall, etc.)

    """
    assert (
        set(mask_name_to_min_amount.keys()).issubset(desired_mask_names)
    ), "mask_name_to_min_amount is making a requirement on a mask you did not request.  Not allowed"

    assert (
        set(mask_name_to_max_amount.keys()).issubset(desired_mask_names)
    ), "mask_name_to_max_amount is making a requirement on a mask you did not request.  Not allowed"

    possible_mask_encoding_types = ["alpha_channel_of_rgba", "mask_is_anything_not_pure_black"]
    assert (
        mask_encoding_type in possible_mask_encoding_types
    ), f"mask_encoding_type must be amongst {possible_mask_encoding_types}, not {mask_encoding_type}"

    possible_mask_names = ["nonfloor", "mainrectangle", "adwall", "foreground", "grass", "relevant", "nonlane", "relevant_lane", "avg_rhs_lane_mask", "players", "inbounds"]
    max_iters = 100000

    mask_name_to_mask_path = annotated_image["mask_name_to_mask_path"]

    assert isinstance(mask_name_to_min_amount, dict)
    assert set(mask_name_to_min_amount.keys()).issubset(possible_mask_names)

    assert isinstance(mask_name_to_max_amount, dict)
    assert set(mask_name_to_max_amount.keys()).issubset(possible_mask_names)

    assert set(desired_mask_names).issubset(possible_mask_names)

    original_hwc_uint8_np = get_hwc_np_uint8_from_image_path_bw(
        image_path=annotated_image['image_path'])
    height, width = original_hwc_uint8_np.shape

    mask_name_to_fullsized_mask = dict()
    for mask_name in desired_mask_names:
        mask_name_to_fullsized_mask[mask_name] = np.zeros(shape=(height, width), dtype=np.uint8)

        mask_path = mask_name_to_mask_path[mask_name]
        # we need to have Blender do the emission trick to make this reliable, the black-pentagoned soccer ball does not have holes in it!
        if mask_encoding_type == "mask_is_anything_not_pure_black":
            segmentation_hwc_uint8_np = get_hwc_np_uint8_from_image_path(
                image_path=mask_path
            )

            mask_name_to_fullsized_mask[mask_name][:, :] = np.max(segmentation_hwc_uint8_np, axis=2) >= 2

        elif mask_encoding_type == "alpha_channel_of_rgba":
            segmentation_hwc_uint8_np = np.array(
                PIL.Image.open(str(mask_path))
            )  # we wish that this only took values 0 or 255

            mask_name_to_fullsized_mask[mask_name][:, :] = (segmentation_hwc_uint8_np[:, :, 3] > 128)

    cropped_originals = np.zeros(shape=(how_many_croppings_to_cut_out, crop_height, crop_width), dtype=np.uint8)
    mask_name_to_cropped_masks = dict()
    for mask_name in desired_mask_names:
        mask_name_to_cropped_masks[mask_name] = np.zeros(shape=(how_many_croppings_to_cut_out, crop_height, crop_width), dtype=np.uint8)

    for mask_name in desired_mask_names:
        num_pixels_on_in_mask = np.sum(mask_name_to_fullsized_mask[mask_name])
        # print(f"num pixels on on mask {num_pixels_on_in_mask}")
        assert (
            num_pixels_on_in_mask > 400
        ), f"image {annotated_image['image_path']} has almost no {mask_name}: only {num_pixels_on_in_mask} pixels are on in it."

    possible = True  # sometimes you ask for something that cannot be satisfied or is unlikely to be satisfied within maxiters

    for cntr in range(how_many_croppings_to_cut_out):
        iters = 0
        while True:
            # choose a cropping at random
            i0 = np.random.randint(0, height - crop_height + 1)
            j0 = np.random.randint(0, width - crop_width + 1)

            # determine if it is acceptable, i.e. if we have at least the minimal amount for all masks
            acceptable = True
            for mask_name in desired_mask_names:  
                s = np.sum(mask_name_to_fullsized_mask[mask_name][i0:i0 + crop_height, j0:j0 + crop_width])
                if (
                    s < mask_name_to_min_amount.get(mask_name, 0)
                    or
                    s > mask_name_to_max_amount.get(mask_name, crop_height * crop_width)
                ):
                    acceptable = False
                    break

            if acceptable:  # found a good i0, j0 that specifies a cropping meeting the requirements
                break

            iters += 1
            if iters > max_iters:
                possible = False
                break

        if not possible:
            return dict(
                success=False,
                num_croppings=0,
                message=f"You might check out image {annotated_image}, it doesnt seem easy to find croppings satisfying your requirements"
            )
        cropped_originals[cntr, :, :] = \
            original_hwc_uint8_np[i0: i0 + crop_height, j0: j0 + crop_width]

        for mask_name in desired_mask_names:
            mask_name_to_cropped_masks[mask_name][cntr, :, :] = \
                mask_name_to_fullsized_mask[mask_name][i0: i0 + crop_height, j0: j0 + crop_width]

    return dict(
        success=True,
        num_croppings=how_many_croppings_to_cut_out,
        cropped_originals=cropped_originals,
        mask_name_to_cropped_masks=mask_name_to_cropped_masks
    )