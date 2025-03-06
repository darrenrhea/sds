from get_mask_path_from_original_path import (
     get_mask_path_from_original_path
)
import argparse
from pathlib import Path
import textwrap


def make_cutouts():  # TODO: cli_tool_ify

    # TODO: See extract_cutouts_by_connected_components for a better way
        
    argp = argparse.ArgumentParser(
        description=textwrap.dedent(
            """\
            Make cutouts of people from existing floor_not_floor or important_people segmentations
            Path to the folder containing either important_people segmentations (that is ideal)
            or floor_not_floor segmentations (that is not ideal, but you will get luckily their background is all floor often)
            and a temporary directory to save the cutouts to.

            You are going to have to manually check the cutouts to make sure they are good,
            so delete all the bad ones.
            """
        ),
        usage=textwrap.dedent(
            """\
            Example:
            
            python ~/r/connected_components/make_cutouts.py \\
            small \\
            ~/r/chicago4k/.flat \\
            ~/r/brewcub_cutouts_approved/baseballs
            """
        )
    )


    argp.add_argument(
        "desired",
        type=str,
        help=textwrap.dedent(
            """\
            Do you want big or small connected components?
            """
        )
    )

    argp.add_argument(
        "segmentation_annotations_folder",
        type=str,
        help=textwrap.dedent(
            """\
            a folder of originals and corresponding masks which is of either important_people or floor_not_floor segmentation convention
            """
        )
    )
    argp.add_argument(
        "out_dir_path",
        type=str,
        help=textwrap.dedent(
            """\
            a folder to save the cutouts into.
            """
        )
    )

    opt = argp.parse_args()
    desired = opt.desired
    
    assert (
        desired in ["big", "small"]
    ), f"ERROR: {desired=} must be either 'big' or 'small'."

    segmentation_annotations_folder = Path(opt.segmentation_annotations_folder).resolve()
    out_dir_path = Path(opt.out_dir_path).resolve()

    assert segmentation_annotations_folder.is_dir(), f"ERROR: {segmentation_annotations_folder} is not a directory."


    out_dir_path.mkdir(exist_ok=True, parents=False)
    original_paths = sorted([x for x in segmentation_annotations_folder.glob("*.jpg")])

    for index, original_path in enumerate(original_paths):
        mask_path = get_mask_path_from_original_path(
            original_path=original_path,
        )
        print(index)
        save_connected_components_of_segmentation_annotation_to_disk(
            original_path=original_path,
            mask_path=mask_path,
            out_dir_path=out_dir_path,
            desired=desired,
        )

if __name__ == "__main__":
    make_cutouts()