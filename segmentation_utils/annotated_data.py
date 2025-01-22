"""
This is used to find training images with masks.
"""

from pathlib import Path
import pprint as pp
from collections import defaultdict



def get_image_ids_in_directory(
    directory
):
    """
    Historically we have directories
    where the original images are {image_id}_color.png
    But we want to move to allow directories which contain
    the originals as {image_id}.jpg
    """
    assert isinstance(directory, Path)
    set_of_ids = set()
    for p in directory.iterdir():
        
        if p.suffix == ".jpg":
            set_of_ids.add(p.stem)
            continue
        elif p.suffix == ".png" and p.stem.endswith("_color"):
           set_of_ids.add(
               p.stem[
                   :
                   -(len("_color"))
                ]
            )
           continue
    return set_of_ids

    

def get_list_of_annotated_images_for_nba(
    must_have_these_masks
):
    """
    Returns a list of dicts where each dictionary points to an original (possibly synthetic)
    color image as well as a json file describing where the landmarks are,
    and possibly various segmentation images as well.
    Here is an example invocation yo:
    ```python
        get_list_of_annotated_images_for_nba(
            must_have_these_masks=[
                "nonfloor",
                "relevant"
            ]
        )
    ```

    """
    clip_names = ["stapleslahou1", "stapleslahou602"]
    possible_mask_names = ["nonfloor", "mainrectangle", "color", "relevant"]
    bad_ids = []

    image_id_to_info = dict()  # first, we gather all distinct image_id_strs:

    for clip_name in clip_names:
        directory_of_images = Path(f"~/r/{clip_name}").expanduser()
        print(f"Getting all image_ids in {directory_of_images}")
        for p in directory_of_images.iterdir():
            if p.suffix != ".png":
                continue
            name = p.stem
            if not name.startswith(f"{clip_name}_"):
                continue
            parts = name.split("_")
            assert parts[0] == clip_name, f"png file {p} must start with {clip_name}"

            assert (
                set(parts[1]).issubset(set("0123456789"))
            ), f"png file {p} does not contain an all-numeric 2nd segment when it was split on underscore.  This is not allowed"

            assert (
                parts[2] in possible_mask_names
            ), f"The  3rd segment of png file {p} when it is split on underscore must be color or nonfloor or mainrectangle or relevant"

            image_id_str = parts[0] + "_" + parts[1]

            image_id_to_info[image_id_str] = dict(
                directory=directory_of_images
            )

    print(f"There are {len(image_id_to_info.keys())} candidates")
    
    image_ids = sorted(list(image_id_to_info.keys()))
    pp.pprint(image_ids)

    list_of_annotated_images = []
    for image_id_str in image_ids:
        if image_id_str in bad_ids:  # for reasons, some images might need to be skipped
            continue
        directory_of_images = image_id_to_info[image_id_str]["directory"]
        original_path = directory_of_images / f"{image_id_str}_color.png"

        if not original_path.is_file():
            continue

        acceptable = True
        for mask_name in must_have_these_masks:
            mask_path = directory_of_images / f"{image_id_str}_{mask_name}.png"
            if not mask_path.is_file():
                acceptable = False
                break

        if not acceptable:
            continue

        dct = dict(
            image_path=original_path,
            image_id_str=image_id_str
        )

        mask_name_to_mask_path = dict()
        for mask_name in must_have_these_masks:
            mask_path = directory_of_images / f"{image_id_str}_{mask_name}.png"
            mask_name_to_mask_path[mask_name] = mask_path

        dct["mask_name_to_mask_path"] = mask_name_to_mask_path

        list_of_annotated_images.append(dct)
    print(f"found {len(list_of_annotated_images)} that had {must_have_these_masks + ['color']}")
    return list_of_annotated_images

def get_list_of_annotated_images_for_gsw(
    must_have_these_masks
):
    """
    Returns a list of dicts where each dictionary points to an original (possibly synthetic)
    color image as well as a json file describing where the landmarks are,
    and possibly various segmentation images as well.
    Here is an example invocation yo:
    ```python
        get_list_of_annotated_images_for_gsw(
            must_have_these_masks=[
                "nonfloor",
                "relevant"
            ]
        )
    ```

    """
    # clip_names = ["final_gsw1/nonfloor_segmentation_downsampled_bw"]
    # clip_names = ["final_gsw1/led_segmentation"]
    clip_names = ["final_gsw1/nonfloor_segmentation_downsampled_one_third"]
    possible_mask_names = ["nonlane", "color", "relevant_lane", "nonfloor", "inbounds", "floor"]
    bad_ids = []

    image_id_to_info = dict()  # first, we gather all distinct image_id_strs:

    for clip_name in clip_names:
        directory_of_images = Path(f"~/r/{clip_name}").expanduser()
        print(f"Getting all image_ids in {directory_of_images}")
        for p in directory_of_images.iterdir():
            # print(p)
            if p.suffix != ".png":
                continue
            name = p.stem
            # print(name)
            # if not name.startswith(f"{clip_name}_"):
            #     continue
            parts = name.split("_")
            # assert parts[0] == clip_name, f"png file {p} must start with {clip_name}"

            # assert (
            #     set(parts[1]).issubset(set("0123456789"))
            # ), f"png file {p} does not contain an all-numeric 2nd segment when it was split on underscore.  This is not allowed"

            # assert (
            #     parts[2] in possible_mask_names
            # ), f"The  3rd segment of png file {p} when it is split on underscore must be color or nonfloor or mainrectangle or relevant"

            image_id_str = parts[0] + "_" + parts[1]

            image_id_to_info[image_id_str] = dict(
                directory=directory_of_images
            )

    print(f"There are {len(image_id_to_info.keys())} candidates")
    
    image_ids = sorted(list(image_id_to_info.keys()))
    # pp.pprint(image_ids)

    list_of_annotated_images = []
    for image_id_str in image_ids:
        if image_id_str in bad_ids:  # for reasons, some images might need to be skipped
            continue
        directory_of_images = image_id_to_info[image_id_str]["directory"]
        original_path = directory_of_images / f"{image_id_str}_color.png"

        if not original_path.is_file():
            continue

        acceptable = True
        for mask_name in must_have_these_masks:
            mask_path = directory_of_images / f"{image_id_str}_{mask_name}.png"
            if not mask_path.is_file():
                acceptable = False
                break

        if not acceptable:
            continue

        dct = dict(
            image_path=original_path,
            image_id_str=image_id_str
        )

        mask_name_to_mask_path = dict()
        for mask_name in must_have_these_masks:
            mask_path = directory_of_images / f"{image_id_str}_{mask_name}.png"
            mask_name_to_mask_path[mask_name] = mask_path

        dct["mask_name_to_mask_path"] = mask_name_to_mask_path

        list_of_annotated_images.append(dct)
    print(f"found {len(list_of_annotated_images)} that had {must_have_these_masks + ['color']}")
    return list_of_annotated_images

def get_list_of_annotated_images_for_den_gsw(
    must_have_these_masks
):
    """
    Returns a list of dicts where each dictionary points to an original (possibly synthetic)
    color image as well as a json file describing where the landmarks are,
    and possibly various segmentation images as well.
    Here is an example invocation yo:
    ```python
        get_list_of_annotated_images_for_den_gsw(
            must_have_these_masks=[
                "nonfloor",
                "relevant"
            ]
        )
    ```

    """
    clip_names = ["den1_gsw1/nonfloor_segmentation"]
    possible_mask_names = ["nonlane", "color", "relevant_lane", "nonfloor", "inbounds", "floor"]
    bad_ids = []

    image_id_to_info = dict()  # first, we gather all distinct image_id_strs:

    for clip_name in clip_names:
        directory_of_images = Path(f"~/r/{clip_name}").expanduser()
        print(f"Getting all image_ids in {directory_of_images}")
        for p in directory_of_images.iterdir():
            # print(p)
            if p.suffix != ".png":
                continue
            name = p.stem
            # print(name)
            # if not name.startswith(f"{clip_name}_"):
            #     continue
            parts = name.split("_")
            # assert parts[0] == clip_name, f"png file {p} must start with {clip_name}"

            # assert (
            #     set(parts[1]).issubset(set("0123456789"))
            # ), f"png file {p} does not contain an all-numeric 2nd segment when it was split on underscore.  This is not allowed"

            # assert (
            #     parts[2] in possible_mask_names
            # ), f"The  3rd segment of png file {p} when it is split on underscore must be color or nonfloor or mainrectangle or relevant"

            image_id_str = parts[0] + "_" + parts[1]

            image_id_to_info[image_id_str] = dict(
                directory=directory_of_images
            )

    print(f"There are {len(image_id_to_info.keys())} candidates")
    
    image_ids = sorted(list(image_id_to_info.keys()))
    # pp.pprint(image_ids)

    list_of_annotated_images = []
    for image_id_str in image_ids:
        if image_id_str in bad_ids:  # for reasons, some images might need to be skipped
            continue
        directory_of_images = image_id_to_info[image_id_str]["directory"]
        original_path = directory_of_images / f"{image_id_str}_color.png"

        if not original_path.is_file():
            continue

        acceptable = True
        for mask_name in must_have_these_masks:
            mask_path = directory_of_images / f"{image_id_str}_{mask_name}.png"
            if not mask_path.is_file():
                acceptable = False
                break

        if not acceptable:
            continue

        dct = dict(
            image_path=original_path,
            image_id_str=image_id_str
        )

        mask_name_to_mask_path = dict()
        for mask_name in must_have_these_masks:
            mask_path = directory_of_images / f"{image_id_str}_{mask_name}.png"
            mask_name_to_mask_path[mask_name] = mask_path

        dct["mask_name_to_mask_path"] = mask_name_to_mask_path

        list_of_annotated_images.append(dct)
    print(f"found {len(list_of_annotated_images)} that had {must_have_these_masks + ['color']}")
    return list_of_annotated_images

def get_list_of_annotated_images_for_den_gsw_ind(
    must_have_these_masks
):
    """
    Returns a list of dicts where each dictionary points to an original (possibly synthetic)
    color image as well as a json file describing where the landmarks are,
    and possibly various segmentation images as well.
    Here is an example invocation yo:
    ```python
        get_list_of_annotated_images_for_den_gsw_ind(
            must_have_these_masks=[
                "nonfloor",
                "relevant"
            ]
        )
    ```

    """
    clip_names = ["den1_gsw1_ind1/nonfloor_segmentation"]
    possible_mask_names = ["nonlane", "color", "relevant_lane", "nonfloor", "inbounds", "floor"]
    bad_ids = []

    image_id_to_info = dict()  # first, we gather all distinct image_id_strs:

    for clip_name in clip_names:
        directory_of_images = Path(f"~/r/{clip_name}").expanduser()
        print(f"Getting all image_ids in {directory_of_images}")
        for p in directory_of_images.iterdir():
            # print(p)
            if p.suffix != ".png":
                continue
            name = p.stem
            # print(name)
            # if not name.startswith(f"{clip_name}_"):
            #     continue
            parts = name.split("_")
            # assert parts[0] == clip_name, f"png file {p} must start with {clip_name}"

            # assert (
            #     set(parts[1]).issubset(set("0123456789"))
            # ), f"png file {p} does not contain an all-numeric 2nd segment when it was split on underscore.  This is not allowed"

            # assert (
            #     parts[2] in possible_mask_names
            # ), f"The  3rd segment of png file {p} when it is split on underscore must be color or nonfloor or mainrectangle or relevant"

            image_id_str = parts[0] + "_" + parts[1]

            image_id_to_info[image_id_str] = dict(
                directory=directory_of_images
            )

    print(f"There are {len(image_id_to_info.keys())} candidates")
    
    image_ids = sorted(list(image_id_to_info.keys()))
    # pp.pprint(image_ids)

    list_of_annotated_images = []
    for image_id_str in image_ids:
        if image_id_str in bad_ids:  # for reasons, some images might need to be skipped
            continue
        directory_of_images = image_id_to_info[image_id_str]["directory"]
        original_path = directory_of_images / f"{image_id_str}_color.png"

        if not original_path.is_file():
            continue

        acceptable = True
        for mask_name in must_have_these_masks:
            mask_path = directory_of_images / f"{image_id_str}_{mask_name}.png"
            if not mask_path.is_file():
                acceptable = False
                break

        if not acceptable:
            continue

        dct = dict(
            image_path=original_path,
            image_id_str=image_id_str
        )

        mask_name_to_mask_path = dict()
        for mask_name in must_have_these_masks:
            mask_path = directory_of_images / f"{image_id_str}_{mask_name}.png"
            mask_name_to_mask_path[mask_name] = mask_path

        dct["mask_name_to_mask_path"] = mask_name_to_mask_path

        list_of_annotated_images.append(dct)
    print(f"found {len(list_of_annotated_images)} that had {must_have_these_masks + ['color']}")
    return list_of_annotated_images

def get_list_of_annotated_images_for_den_gsw_ind_okst(
    must_have_these_masks
):
    """
    Returns a list of dicts where each dictionary points to an original (possibly synthetic)
    color image as well as a json file describing where the landmarks are,
    and possibly various segmentation images as well.
    Here is an example invocation yo:
    ```python
        get_list_of_annotated_images_for_den_gsw_ind_okst(
            must_have_these_masks=[
                "nonfloor",
                "relevant"
            ]
        )
    ```

    """
    clip_names = ["den1_gsw1_ind1_okstfull/nonfloor_segmentation"]
    possible_mask_names = ["nonlane", "color", "relevant_lane", "nonfloor", "inbounds", "floor"]
    bad_ids = []

    image_id_to_info = dict()  # first, we gather all distinct image_id_strs:

    for clip_name in clip_names:
        directory_of_images = Path(f"~/r/{clip_name}").expanduser()
        print(f"Getting all image_ids in {directory_of_images}")
        for p in directory_of_images.iterdir():
            # print(p)
            if p.suffix != ".png":
                continue
            name = p.stem
            # print(name)
            # if not name.startswith(f"{clip_name}_"):
            #     continue
            parts = name.split("_")
            # assert parts[0] == clip_name, f"png file {p} must start with {clip_name}"

            # assert (
            #     set(parts[1]).issubset(set("0123456789"))
            # ), f"png file {p} does not contain an all-numeric 2nd segment when it was split on underscore.  This is not allowed"

            # assert (
            #     parts[2] in possible_mask_names
            # ), f"The  3rd segment of png file {p} when it is split on underscore must be color or nonfloor or mainrectangle or relevant"

            image_id_str = parts[0] + "_" + parts[1]

            image_id_to_info[image_id_str] = dict(
                directory=directory_of_images
            )

    print(f"There are {len(image_id_to_info.keys())} candidates")
    
    image_ids = sorted(list(image_id_to_info.keys()))
    # pp.pprint(image_ids)

    list_of_annotated_images = []
    for image_id_str in image_ids:
        if image_id_str in bad_ids:  # for reasons, some images might need to be skipped
            continue
        directory_of_images = image_id_to_info[image_id_str]["directory"]
        original_path = directory_of_images / f"{image_id_str}_color.png"

        if not original_path.is_file():
            continue

        acceptable = True
        for mask_name in must_have_these_masks:
            mask_path = directory_of_images / f"{image_id_str}_{mask_name}.png"
            if not mask_path.is_file():
                acceptable = False
                break

        if not acceptable:
            continue

        dct = dict(
            image_path=original_path,
            image_id_str=image_id_str
        )

        mask_name_to_mask_path = dict()
        for mask_name in must_have_these_masks:
            mask_path = directory_of_images / f"{image_id_str}_{mask_name}.png"
            mask_name_to_mask_path[mask_name] = mask_path

        dct["mask_name_to_mask_path"] = mask_name_to_mask_path

        list_of_annotated_images.append(dct)
    print(f"found {len(list_of_annotated_images)} that had {must_have_these_masks + ['color']}")
    return list_of_annotated_images

def get_list_of_annotated_images_for_okstfull(
    must_have_these_masks
):
    """
    Returns a list of dicts where each dictionary points to an original (possibly synthetic)
    color image as well as a json file describing where the landmarks are,
    and possibly various segmentation images as well.
    Here is an example invocation yo:
    ```python
        get_list_of_annotated_images_for_okstfull(
            must_have_these_masks=[
                "nonfloor",
                "relevant"
            ]
        )
    ```

    """
    clip_names = ["okstfull/nonfloor_segmentation"]
    possible_mask_names = ["nonlane", "color", "relevant_lane", "nonfloor", "inbounds", "floor"]
    bad_ids = []

    image_id_to_info = dict()  # first, we gather all distinct image_id_strs:

    for clip_name in clip_names:
        directory_of_images = Path(f"~/r/{clip_name}").expanduser()
        print(f"Getting all image_ids in {directory_of_images}")
        for p in directory_of_images.iterdir():
            # print(p)
            if p.suffix != ".png":
                continue
            name = p.stem
            # print(name)
            # if not name.startswith(f"{clip_name}_"):
            #     continue
            parts = name.split("_")
            # assert parts[0] == clip_name, f"png file {p} must start with {clip_name}"

            # assert (
            #     set(parts[1]).issubset(set("0123456789"))
            # ), f"png file {p} does not contain an all-numeric 2nd segment when it was split on underscore.  This is not allowed"

            # assert (
            #     parts[2] in possible_mask_names
            # ), f"The  3rd segment of png file {p} when it is split on underscore must be color or nonfloor or mainrectangle or relevant"

            image_id_str = parts[0] + "_" + parts[1]

            image_id_to_info[image_id_str] = dict(
                directory=directory_of_images
            )

    print(f"There are {len(image_id_to_info.keys())} candidates")
    
    image_ids = sorted(list(image_id_to_info.keys()))
    # pp.pprint(image_ids)

    list_of_annotated_images = []
    for image_id_str in image_ids:
        if image_id_str in bad_ids:  # for reasons, some images might need to be skipped
            continue
        directory_of_images = image_id_to_info[image_id_str]["directory"]
        original_path = directory_of_images / f"{image_id_str}_color.png"

        if not original_path.is_file():
            continue

        acceptable = True
        for mask_name in must_have_these_masks:
            mask_path = directory_of_images / f"{image_id_str}_{mask_name}.png"
            if not mask_path.is_file():
                acceptable = False
                break

        if not acceptable:
            continue

        dct = dict(
            image_path=original_path,
            image_id_str=image_id_str
        )

        mask_name_to_mask_path = dict()
        for mask_name in must_have_these_masks:
            mask_path = directory_of_images / f"{image_id_str}_{mask_name}.png"
            mask_name_to_mask_path[mask_name] = mask_path

        dct["mask_name_to_mask_path"] = mask_name_to_mask_path

        list_of_annotated_images.append(dct)
    print(f"found {len(list_of_annotated_images)} that had {must_have_these_masks + ['color']}")
    return list_of_annotated_images


def get_list_of_annotated_images_for_den(
    must_have_these_masks
):
    """
    Returns a list of dicts where each dictionary points to an original (possibly synthetic)
    color image as well as a json file describing where the landmarks are,
    and possibly various segmentation images as well.
    Here is an example invocation yo:
    ```python
        get_list_of_annotated_images_for_den(
            must_have_these_masks=[
                "nonfloor",
                "relevant"
            ]
        )
    ```

    """
    clip_names = ["den1/nonfloor_segmentation"]
    possible_mask_names = ["nonlane", "color", "relevant_lane", "nonfloor", "inbounds", "floor"]
    bad_ids = []

    image_id_to_info = dict()  # first, we gather all distinct image_id_strs:

    for clip_name in clip_names:
        directory_of_images = Path(f"~/r/{clip_name}").expanduser()
        print(f"Getting all image_ids in {directory_of_images}")
        for p in directory_of_images.iterdir():
            # print(p)
            # if p.suffix != ".png":
            #     continue
            name = p.stem
            # print(name)
            # if not name.startswith(f"{clip_name}_"):
            #     continue
            parts = name.split("_")
            # assert parts[0] == clip_name, f"png file {p} must start with {clip_name}"

            # assert (
            #     set(parts[1]).issubset(set("0123456789"))
            # ), f"png file {p} does not contain an all-numeric 2nd segment when it was split on underscore.  This is not allowed"

            # assert (
            #     parts[2] in possible_mask_names
            # ), f"The  3rd segment of png file {p} when it is split on underscore must be color or nonfloor or mainrectangle or relevant"

            image_id_str = parts[0] + "_" + parts[1]

            image_id_to_info[image_id_str] = dict(
                directory=directory_of_images
            )

    print(f"There are {len(image_id_to_info.keys())} candidates")
    
    image_ids = sorted(list(image_id_to_info.keys()))
    # pp.pprint(image_ids)

    list_of_annotated_images = []
    for image_id_str in image_ids:
        if image_id_str in bad_ids:  # for reasons, some images might need to be skipped
            continue
        directory_of_images = image_id_to_info[image_id_str]["directory"]
        original_path = directory_of_images / f"{image_id_str}_color.jpg"

        if not original_path.is_file():
            continue

        acceptable = True
        for mask_name in must_have_these_masks:
            mask_path = directory_of_images / f"{image_id_str}_{mask_name}.png"
            if not mask_path.is_file():
                acceptable = False
                break

        if not acceptable:
            continue

        dct = dict(
            image_path=original_path,
            image_id_str=image_id_str
        )

        mask_name_to_mask_path = dict()
        for mask_name in must_have_these_masks:
            mask_path = directory_of_images / f"{image_id_str}_{mask_name}.png"
            mask_name_to_mask_path[mask_name] = mask_path

        dct["mask_name_to_mask_path"] = mask_name_to_mask_path

        list_of_annotated_images.append(dct)
    print(f"found {len(list_of_annotated_images)} that had {must_have_these_masks + ['color']}")
    return list_of_annotated_images

def get_list_of_annotated_images_for_gsw_fake(
    must_have_these_masks
):
    """
    Returns a list of dicts where each dictionary points to an original (possibly synthetic)
    color image as well as a json file describing where the landmarks are,
    and possibly various segmentation images as well.
    Here is an example invocation yo:
    ```python
        get_list_of_annotated_images_for_gsw_fake(
            must_have_these_masks=[
                "players",
                "avg_rhs_lane_mask"
            ]
        )
    ```

    """
    clip_names = ["~/awecom/data/clips/gsw1/fake_segmentation_data"]
    possible_mask_names = ["frame", "players", "avg_rhs_lane_mask"]
    bad_ids = []

    image_id_to_info = dict()  # first, we gather all distinct image_id_strs:

    for clip_name in clip_names:
        directory_of_images = Path(f"{clip_name}").expanduser()
        print(f"Getting all image_ids in {directory_of_images}")
        for p in directory_of_images.iterdir():
            # print(p)
            if p.suffix != ".png":
                continue
            name = p.stem
            parts = name.split("_")
            image_id_str = parts[0]
            image_id_to_info[image_id_str] = dict(
                directory=directory_of_images
            )

    print(f"There are {len(image_id_to_info.keys())} candidates")
    
    image_ids = sorted(list(image_id_to_info.keys()))
    # pp.pprint(image_ids)

    directory_of_relevance = Path(f"~/r/segmentation_utils").expanduser()
    list_of_annotated_images = []
    for image_id_str in image_ids:
        if image_id_str in bad_ids:  # for reasons, some images might need to be skipped
            continue
        # print(f"image id string {image_id_str}")
        directory_of_images = image_id_to_info[image_id_str]["directory"]
        original_path = directory_of_images / f"{image_id_str}_frame.png"
        
        # print(f"original path {original_path}")

        if not original_path.is_file():
            continue

        acceptable = True
        for mask_name in must_have_these_masks:
            if mask_name == "avg_rhs_lane_mask":
                mask_path = directory_of_relevance / f"{mask_name}.png"
            else:
                mask_path = directory_of_images / f"{image_id_str}_{mask_name}.png"
            if not mask_path.is_file():
                acceptable = False
                break

        if not acceptable:
            continue

        dct = dict(
            image_path=original_path,
            image_id_str=image_id_str
        )

        mask_name_to_mask_path = dict()
        for mask_name in must_have_these_masks:
            if mask_name == "avg_rhs_lane_mask":
                mask_path = directory_of_relevance / f"{mask_name}.png"
            else:
                mask_path = directory_of_images / f"{image_id_str}_{mask_name}.png"
            mask_name_to_mask_path[mask_name] = mask_path

        dct["mask_name_to_mask_path"] = mask_name_to_mask_path

        list_of_annotated_images.append(dct)
    print(f"found {len(list_of_annotated_images)} that had {must_have_these_masks + ['frame']}")
    return list_of_annotated_images


def get_list_of_annotated_images_for_ncaa_kansas(
    must_have_these_masks
):
    """
    Returns a list of dicts where each dictionary points to an original (possibly synthetic)
    color image as well as a json file describing where the landmarks are,
    and possibly various segmentation images as well.
    Here is an example invocation yo:
    ```python
        get_list_of_annotated_images_for_nba(
            must_have_these_masks=[
                "nonfloor",
                "relevant"
            ]
        )
    ```

    """
    clip_names = ["swinney1"]
    possible_mask_names = ["nonfloor", "color", "relevant"]
    bad_ids = []

    image_id_to_info = dict()  # first, we gather all distinct image_id_strs:

    for clip_name in clip_names:
        directory_of_images = Path(f"~/r/{clip_name}").expanduser()
        print(f"Getting all image_ids in {directory_of_images}")
        for p in directory_of_images.iterdir():
            if p.suffix != ".png":
                continue
            name = p.stem
            if not name.startswith(f"{clip_name}_"):
                continue
            parts = name.split("_")
            assert parts[0] == clip_name, f"png file {p} must start with {clip_name}"

            assert (
                set(parts[1]).issubset(set("0123456789"))
            ), f"png file {p} does not contain an all-numeric 2nd segment when it was split on underscore.  This is not allowed"

            assert (
                parts[2] in possible_mask_names
            ), f"The  3rd segment of png file {p} when it is split on underscore must be color or nonfloor or mainrectangle or relevant"

            image_id_str = parts[0] + "_" + parts[1]

            image_id_to_info[image_id_str] = dict(
                directory=directory_of_images
            )

    print(f"There are {len(image_id_to_info.keys())} candidates")
    
    image_ids = sorted(list(image_id_to_info.keys()))
    pp.pprint(image_ids)

    list_of_annotated_images = []
    for image_id_str in image_ids:
        if image_id_str in bad_ids:  # for reasons, some images might need to be skipped
            continue
        directory_of_images = image_id_to_info[image_id_str]["directory"]
        original_path = directory_of_images / f"{image_id_str}_color.png"

        if not original_path.is_file():
            continue

        acceptable = True
        for mask_name in must_have_these_masks:
            mask_path = directory_of_images / f"{image_id_str}_{mask_name}.png"
            if not mask_path.is_file():
                acceptable = False
                break

        if not acceptable:
            continue

        dct = dict(
            image_path=original_path,
            image_id_str=image_id_str
        )

        mask_name_to_mask_path = dict()
        for mask_name in must_have_these_masks:
            mask_path = directory_of_images / f"{image_id_str}_{mask_name}.png"
            mask_name_to_mask_path[mask_name] = mask_path

        dct["mask_name_to_mask_path"] = mask_name_to_mask_path

        list_of_annotated_images.append(dct)
    print(f"found {len(list_of_annotated_images)} that had {must_have_these_masks + ['color']}")
    return list_of_annotated_images




def get_list_of_annotated_images_for_ncaa_basketball(
    must_have_these_masks
):
    """
    Returns a list of dicts where each dct points to an original (possibly synthetic)
    color image as well as a json file describing where the landmarks are,
    and possibly various segmentation images as well.
    Here is an example invokation yo:
    ```python
        get_list_of_annotated_images_for_basketball(
            must_have_these_masks=["nonfloor"]
        )
    ```

    """
    possible_mask_names = ["nonfloor", "mainrectangle", "relevant"] + ["color"]
    bad_ids = []
    directory_of_images = Path("~/r/swinney").expanduser()
    print(f"Getting all images in {directory_of_images}")
    image_ids = set()  # first, we gather all distinct image_id_strs:
    for p in directory_of_images.iterdir():
        if p.suffix != ".png":
            continue
        name = p.stem
        if not name.startswith("rice1_"):
            continue
        parts = name.split("_")
        assert parts[0] == "rice1", f"png file {p} must start with rice1_"

        assert (
            set(parts[1]).issubset(set("0123456789"))
        ), f"png file {p} does not contain an all-numeric 2nd segment when it was split on underscore.  This is not allowed"

        assert (
            parts[2] in possible_mask_names
        ), f"The  3rd segment of png file {p} when it is split on underscore must be color or nonfloor"

        image_id_str = parts[0] + "_" + parts[1]

        image_ids.add(image_id_str)

    print(f"There are {len(image_ids)} candidates, namely:")
    image_ids = sorted(list(image_ids))
    
    list_of_annotated_images = []
    for image_id_str in image_ids:
        if image_id_str in bad_ids:  # for reasons, some images might need to be skipped
            continue
        original_path = directory_of_images / f"{image_id_str}_color.png"

        if not original_path.is_file():
            continue

        acceptable = True
        for mask_name in must_have_these_masks:
            mask_path = directory_of_images / f"{image_id_str}_{mask_name}.png"
            if not mask_path.is_file():
                acceptable = False
                break

        if not acceptable:
            continue

        dct = dict(image_path=original_path, image_id_str=image_id_str)

        mask_name_to_mask_path = dict()
        for mask_name in must_have_these_masks:
            mask_path = directory_of_images / f"{image_id_str}_{mask_name}.png"
            mask_name_to_mask_path[mask_name] = mask_path

        dct["mask_name_to_mask_path"] = mask_name_to_mask_path

        list_of_annotated_images.append(dct)
    print(f"found {len(list_of_annotated_images)} that had nonfloor, color, and mainrectangle")
    return list_of_annotated_images

def get_list_of_annotated_images_for_brooklyn(
    must_have_these_masks
):
    """
    Returns a list of dicts where each dictionary points to an original (possibly synthetic)
    color image as well as a json file describing where the landmarks are,
    and possibly various segmentation images as well.
    Here is an example invocation yo:
    ```python
        get_list_of_annotated_images_for_gsw(
            must_have_these_masks=[
                "nonfloor",
                "relevant"
            ]
        )
    ```

    """
    # clip_names = ["brooklyn_nets_barclays_center/nonfloor_segmentation_downsampled_one_half"]
    clip_names = ["brooklyn_nets_barclays_center/nonfloor_segmentation_train"]
    # clip_names = ["brooklyn_nets_barclays_center/nonfloor_segmentation_downsampled_one_third"]
    possible_mask_names = ["nonlane", "color", "relevant_lane", "nonfloor", "inbounds", "floor"]
    bad_ids = []

    image_id_to_info = dict()  # first, we gather all distinct image_id_strs:

    for clip_name in clip_names:
        directory_of_images = Path(f"~/r/{clip_name}").expanduser()
        print(f"Getting all image_ids in {directory_of_images}")
        for p in directory_of_images.iterdir():
            # print(p)
            if p.suffix != ".png":
                continue
            name = p.stem
            # print(name)
            # if not name.startswith(f"{clip_name}_"):
            #     continue
            parts = name.split("_")
            # assert parts[0] == clip_name, f"png file {p} must start with {clip_name}"

            # assert (
            #     set(parts[1]).issubset(set("0123456789"))
            # ), f"png file {p} does not contain an all-numeric 2nd segment when it was split on underscore.  This is not allowed"

            # assert (
            #     parts[2] in possible_mask_names
            # ), f"The  3rd segment of png file {p} when it is split on underscore must be color or nonfloor or mainrectangle or relevant"

            image_id_str = "_".join(parts[:-1])

            image_id_to_info[image_id_str] = dict(
                directory=directory_of_images
            )

    print(f"There are {len(image_id_to_info.keys())} candidates")
    
    image_ids = sorted(list(image_id_to_info.keys()))
    # pp.pprint(image_ids)

    list_of_annotated_images = []
    for image_id_str in image_ids:
        if image_id_str in bad_ids:  # for reasons, some images might need to be skipped
            continue
        directory_of_images = image_id_to_info[image_id_str]["directory"]
        original_path = directory_of_images / f"{image_id_str}_color.png"

        if not original_path.is_file():
            continue

        acceptable = True
        for mask_name in must_have_these_masks:
            mask_path = directory_of_images / f"{image_id_str}_{mask_name}.png"
            if not mask_path.is_file():
                acceptable = False
                break

        if not acceptable:
            continue

        dct = dict(
            image_path=original_path,
            image_id_str=image_id_str
        )

        mask_name_to_mask_path = dict()
        for mask_name in must_have_these_masks:
            mask_path = directory_of_images / f"{image_id_str}_{mask_name}.png"
            mask_name_to_mask_path[mask_name] = mask_path

        dct["mask_name_to_mask_path"] = mask_name_to_mask_path

        list_of_annotated_images.append(dct)
    print(f"found {len(list_of_annotated_images)} that had {must_have_these_masks + ['color']}")
    return list_of_annotated_images

def get_list_of_annotated_images_for_soccer(
    must_have_these_masks
):
    """
    Returns a list of dicts where each dct points to an original (possibly synthetic)
    color image as well as a json file describing where the landmarks are,
    and possibly various segmentation images as well.
    Here is an example invokation yo:
    ```python
        get_list_of_annotated_images(
            use_case_name="soccer",
            must_have_these_masks=["foreground", "grass", "adwall"]
        )
    ```

    """
    bad_ids = []
    directory_of_images = Path("~/synthetic_soccer").expanduser()
    print(f"Getting all images in {directory_of_images}")
    image_ids = set()
    for p in directory_of_images.iterdir():
        name = p.stem
        image_id, _ = name.split("_")
        image_ids.add(image_id)
    print(f"There are {len(image_ids)} candidates")
    image_ids = sorted(list(image_ids))
    list_of_annotated_images = []
    for image_id in image_ids:
        if image_id in bad_ids:  # for reasons, some images might need to be skipped
            continue
        everything = directory_of_images / f"{image_id}_everything.png"
    
        if not everything.is_file():
            continue
   
        acceptable = True
        for mask_name in must_have_these_masks:
            mask_path = directory_of_images / f"{image_id}_{mask_name}.png"
            if not mask_path.is_file():
                acceptable = False
                break
  
        if not acceptable:
            continue

        dct = dict(image_path=everything, image_id_int=int(image_id),  image_id_str=image_id)

        mask_name_to_mask_path = dict()
        for mask_name in must_have_these_masks:
            mask_path = directory_of_images / f"{image_id}_{mask_name}.png"
            mask_name_to_mask_path[mask_name] = mask_path

        dct["mask_name_to_mask_path"] = mask_name_to_mask_path

        list_of_annotated_images.append(dct)

    return list_of_annotated_images




def get_list_of_annotated_images_from_several_directories(
    must_have_these_masks,
    directories_to_gather_from_with_limits
):
    """
    This returns a list of dicts.  What kind of dicts?
    Each dict in the list must have:
    1. the key
        "image_id_str"
        which maps to an annotated_image_id string like
        'BKN_CITY_2021-11-03_PGM_short_002392' ;

    2. the key
        "image_path"
    which maps to an absolute Path
        Path('/home/drhea/r/brooklyn_nets_barclays_center/nonfloor_segmentation/BKN_CITY_2021-11-03_PGM_short_002392_color.png') ;

    3. the key
        "mask_name_to_mask_path"
    mapping to a dictionary like
        {
           'inbounds': Path('/home/drhea/r/brooklyn_nets_barclays_center/nonfloor_segmentation/BKN_CITY_2021-11-03_PGM_short_002392_inbounds.png'),
           'nonfloor': Path('/home/drhea/r/brooklyn_nets_barclays_center/nonfloor_segmentation/BKN_CITY_2021-11-03_PGM_short_002392_nonfloor.png')}
        }
    
    where the keys in that dictionary are the must_have_these_masks images' abs paths.


    Here is an example invocation:
    ```python
        get_list_of_annotated_images_from_several_directories(
            must_have_these_masks=[
                "nonfloor",
                "relevant"
            ],
            directories_to_gather_from_with_limits = [
                (
                    Path(f"~/r/brooklyn_nets_barclays_center/nonfloor_segmentation").expanduser(),
                    10
                ),
                (
                    Path(f"~/r/boston_celtics/nonfloor").expanduser(),
                    20
                )
            ]
        )
    ```

    """

    directory_to_how_many_from_this_directory = defaultdict(int)
    
   
    possible_mask_names = ["nonlane", "color", "relevant_lane", "nonfloor", "inbounds", "floor", "nonled"]
    bad_ids = []

    image_id_to_info = dict()  # first, we gather all distinct image_id_strs:
    valid_clip_names = [
        "BOS_CORE_2022-03-30_MIA_PGM_30Mbps",
        "nets20211116",
        "nets1progfeed",
        "nets20211103",
        "nets20211117",
        "BKN_CITY_2021-11-30_PGM",
        "BKN_CITY_2022-01-03_PGM_short",
        "BKN_CITY_2021-12-16_PGM_short",
        "BKN_CITY_2021-11-03_PGM_short",
        "nets20211130",
        "BKN_CITY_2022-01-26_PGM_short",
        "nets20211203",
        "nets20211204",
        "BKN_CITY_2021-12-18_PGM_short",
        "BKN_CITY_2021-11-17_PGM",
        "nets20211127",
        "PHI_CORE_2022-04-16_TOR_PGM",
        "PHI_CORE_2022-04-18_TOR_PGM",
        "PHI_CORE_2022-03-14_DEN_PGM",

    ]

    limit_for_directory = dict()
    for directory_of_images, limit in directories_to_gather_from_with_limits:
        limit_for_directory[directory_of_images] = limit
        set_of_image_ids = get_image_ids_in_directory(
            directory=directory_of_images
        )
        for image_id_str in sorted(list(set_of_image_ids)):   
            if image_id_str in image_id_to_info:
                raise Exception(
                    f"This is too weird: the image_id {image_id_str} occurs in several directories, namely"
                    f"{directory_of_images} and {image_id_to_info[image_id_str]['directory']}"
                )     
            image_id_to_info[image_id_str] = dict(
                directory=directory_of_images
            )
        

    print(f"There are {len(image_id_to_info.keys())} candidates")

    
    image_ids = sorted(list(image_id_to_info.keys()))
    pp.pprint(image_ids)

    list_of_annotated_images = []
    for image_id_str in image_ids:
        if image_id_str in bad_ids:  # for reasons, some images might need to be skipped
            continue
        directory_of_images = image_id_to_info[image_id_str]["directory"]
        original_path = directory_of_images / f"{image_id_str}_color.png"
        if not original_path.exists():
            original_path = directory_of_images / f"{image_id_str}.jpg"
            if not original_path.exists():
                raise Exception("Neither original path exists?")   

        acceptable = True
        for mask_name in must_have_these_masks:
            mask_path = directory_of_images / f"{image_id_str}_{mask_name}.png"
            if not mask_path.is_file():
                print(f"Candidate {image_id_str} rejected because {mask_path} does not exist")
                acceptable = False
                break

        if not acceptable:
            continue

        if directory_to_how_many_from_this_directory[directory_of_images] >= limit_for_directory[directory_of_images]:
            print(f"refusing to add more than {limit_for_directory[directory_of_images]} from {directory_of_images}")
            continue

        dct = dict(
            image_path=original_path,
            image_id_str=image_id_str
        )

        mask_name_to_mask_path = dict()
        for mask_name in must_have_these_masks:
            mask_path = directory_of_images / f"{image_id_str}_{mask_name}.png"
            mask_name_to_mask_path[mask_name] = mask_path

        dct["mask_name_to_mask_path"] = mask_name_to_mask_path

        list_of_annotated_images.append(dct)
        directory_to_how_many_from_this_directory[directory_of_images] += 1

    print(f"found {len(list_of_annotated_images)}")
    return list_of_annotated_images


def get_list_of_annotated_images(
    use_case_name,
    must_have_these_masks
):
    """
    Returns a list of dicts where each dct points to an original (possibly synthetic)
    color image as well as a json file describing where the landmarks are,
    and possibly various segmentation images as well.
    Here is an example invokation yo:
    ```python
        get_list_of_annotated_images(
            use_case_name="soccer",
            must_have_these_masks=["foreground", "grass", "adwall"]
        )
    ```

    """
    # Dispatch based on the use_case_name:

    if use_case_name == "soccer":
        list_of_annotated_images = get_list_of_annotated_images_for_soccer(
            must_have_these_masks=must_have_these_masks
        )
    elif use_case_name == "nba":
        list_of_annotated_images = get_list_of_annotated_images_for_nba(
            must_have_these_masks=must_have_these_masks
        )
    elif use_case_name == "ncaa_basketball":
        list_of_annotated_images = get_list_of_annotated_images_for_ncaa_basketball(
            must_have_these_masks=must_have_these_masks
        )
    elif use_case_name == "ncaa_kansas":
        list_of_annotated_images = get_list_of_annotated_images_for_ncaa_kansas(
            must_have_these_masks=must_have_these_masks
        )
    elif use_case_name == "gsw":
        list_of_annotated_images = get_list_of_annotated_images_for_gsw(
            must_have_these_masks=must_have_these_masks
        )
    elif use_case_name == "brooklyn":
        list_of_annotated_images = get_list_of_annotated_images_for_brooklyn(
        must_have_these_masks=must_have_these_masks
    )
    elif use_case_name == "gsw_fake":
        list_of_annotated_images = get_list_of_annotated_images_for_gsw_fake(
            must_have_these_masks=must_have_these_masks
        )
        # for i in range(0, 5):
        #     print(list_of_annotated_images[i])
    elif use_case_name == "den":
        list_of_annotated_images = get_list_of_annotated_images_for_den(
            must_have_these_masks=must_have_these_masks
        )
    elif use_case_name == "den_gsw":
        list_of_annotated_images = get_list_of_annotated_images_for_den_gsw(
            must_have_these_masks=must_have_these_masks
        )
    elif use_case_name == "den_gsw_ind":
        list_of_annotated_images = get_list_of_annotated_images_for_den_gsw_ind(
            must_have_these_masks=must_have_these_masks
        )
    elif use_case_name == "den_gsw_ind_okstfull":
        list_of_annotated_images = get_list_of_annotated_images_for_den_gsw_ind_okst(
            must_have_these_masks=must_have_these_masks
        )
    elif use_case_name == "okstfull":
        list_of_annotated_images = get_list_of_annotated_images_for_okstfull(
            must_have_these_masks=must_have_these_masks
        )

    else:
        raise Exception(f"use_case_name {use_case_name} unrecognized")

    assert isinstance(list_of_annotated_images, list)
    for dct in list_of_annotated_images:
        assert "mask_name_to_mask_path" in dct
        mask_name_to_mask_path = dct["mask_name_to_mask_path"]
        assert isinstance(mask_name_to_mask_path, dict)

        for name, p in mask_name_to_mask_path.items():
            assert isinstance(p, Path)
            if not p.is_file():
                print(f"This is not a file {p}")
            assert p.is_file()

        for mn in must_have_these_masks:
            assert mn in mask_name_to_mask_path

        assert "image_path" in dct
        image_path = dct["image_path"]
        assert isinstance(image_path, Path)
        assert image_path.is_file()

    return list_of_annotated_images




if __name__ == "__main__":

    list_of_annotated_images = get_list_of_annotated_images_for_celtics(
        must_have_these_masks=[
            "nonfloor",
            "inbounds"
        ]
    )

    pp.pprint(list_of_annotated_images)