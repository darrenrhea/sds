import sys
from get_repo_ids_to_use_for_fake_data import (
     get_repo_ids_to_use_for_fake_data
)
from get_camera_posed_fake_background_annotations import (
     get_camera_posed_fake_background_annotations
)
import time
from pathlib import Path
import numpy as np
import PIL.Image
from prii import (
     prii
)
from get_valid_cutout_kinds import (
     get_valid_cutout_kinds
)
from better_json import (
     color_print_json
)
from group_cutouts_by_kind import (
     group_cutouts_by_kind
)
from prii_named_xy_points_on_image import (
     prii_named_xy_points_on_image
)
from paste_multiple_cutouts_onto_one_camera_posed_segmentation_annotation import (
     paste_multiple_cutouts_onto_one_camera_posed_segmentation_annotation
)
from get_cutout_augmentation import (
     get_cutout_augmentation
)
from get_cutouts import (
     get_cutouts
)
from get_camera_posed_actual_annotations import (
     get_camera_posed_actual_annotations
)
from make_rgba_from_original_and_mask_paths import (
     make_rgba_from_original_and_mask_paths
)

white_list_for_interwetten = [
    "bay-mta-2024-03-22-mxf_101801",
    "bay-mta-2024-03-22-mxf_102063",
    "bay-mta-2024-03-22-mxf_102183",
    "bay-mta-2024-03-22-mxf_102188",
    "bay-mta-2024-03-22-mxf_102270",
    "bay-mta-2024-03-22-mxf_102281",
    "bay-mta-2024-03-22-mxf_102305",
    "bay-mta-2024-03-22-mxf_102315",
    "bay-mta-2024-03-22-mxf_102335",
    "bay-mta-2024-03-22-mxf_102335",
    "bay-mta-2024-03-22-mxf_191554",
    "bay-mta-2024-03-22-mxf_191757",
]

def make_fake_segmentation_annotations(
    out_dir: Path,
    shard_id: int
):
    """
    This makes a lot of fake annotations
    by pasting multiple cutouts onto camera-posed-background-annotations.
    Those background annotations might be actual or fake,
    for instance they may be fake-led-ads, or human annotations.

    Since the pasted object forced the answer to foreground,
    it should work for floor_not_floor as well as for led_not_led.
    """
    
    fakes_dir = Path(out_dir).resolve()
    # segmentation_convention = "floor_not_floor"
    segmentation_convention = "led_not_led"

    
    # You have to choose how many cutouts of each kind to paste onto an actual annotation to make a fake annotation:


    cutout_kind_to_num_cutouts_to_paste = dict(
        player=5,
        referee=2,
        coach=2,
        ball=1,
    )


    cutout_kind_to_transform = dict(
        player=get_cutout_augmentation("player"),
        referee=get_cutout_augmentation("referee"),
        coach=get_cutout_augmentation("coach"),
        ball=get_cutout_augmentation("ball"),
    )
    
    # choose what albumentations augmentation to use on cutouts, if any:
    # TODO: this should be different per kind of cutout.
    # The black pants people need care.
    

    # How make fake annotations to make out of one real/actual annotation:
    num_fakes_per_actual_annotation = 1
    
    #if segmentation_convention == "floor_not_floor":
    actual_annotations = get_camera_posed_fake_background_annotations(
        subdir_name="fake_backgrounds",
        shard_id=shard_id
    )
    
    # elif segmentation_convention == "led_not_led":
    #     repo_ids_to_use = get_repo_ids_to_use_for_fake_data(
    #         floor_id="munich",
    #         segmentation_convention=segmentation_convention
    #     )
    #     actual_annotations = get_camera_posed_actual_annotations(
    #         repo_ids_to_use=repo_ids_to_use
    #     )

    num_actual_annotations = len(actual_annotations)
    print(f"We are using {num_actual_annotations} actual annotations.")
    how_many_fake_annotations_will_be_made = num_fakes_per_actual_annotation * num_actual_annotations
    print(f"Going to make {how_many_fake_annotations_will_be_made} fake annotations.")
    fake_counter = 0


    start_time = time.time()
    
    cutouts = get_cutouts(
        context_id="athens",
        diminish_for_debugging=False,
    )
    
    stop_time = time.time()
    print(f"Loading {len(cutouts)} cutouts into RAM in {stop_time - start_time} seconds.")

    cutouts_by_kind = group_cutouts_by_kind(
        cutouts=cutouts
    ) 

    print("Here are the cutouts we are going to paste onto the actual annotations:")
    print("You better check them for coverage of all uniform possibilities and adequate variety.")
    valid_cutout_kinds = get_valid_cutout_kinds()
    
    for kind in valid_cutout_kinds:
        print(f"All cutouts of kind {kind}:")
        for cutout in cutouts_by_kind[kind]:
            color_print_json(cutout.metadata)
            prii_named_xy_points_on_image(
                image=cutout.rgba_np_u8,
                name_to_xy=cutout.metadata["name_to_xy"]
            )
    
    # we could precalculate augmentations, or do it on the fly right before pasting:
    # augmented_cutouts = make_augmented_cutouts(cutouts)

    
    fakes_dir.mkdir(exist_ok=True, parents=True)

    for actual_annotation in actual_annotations:  # pick an actual/read segmentation annotation to add cutouts to:
        actual_annotation_original_file_path = actual_annotation["original_file_path"]
        actual_annotation_mask_file_path = actual_annotation["mask_file_path"]
        # annotation_id = actual_annotation["annotation_id"]
        # if annotation_id not in white_list:
        #     continue
        camera_pose = actual_annotation["camera_pose"]
        

        # loading stuff from file is expensive, so load the actual once and reuse it:

        # this crushes 16 bit pngs down to 8 bit, but in the future we might want to keep them 16 bit:
        actual_annotation_rgba_np_u8 = make_rgba_from_original_and_mask_paths(
            original_path=actual_annotation_original_file_path,
            mask_path=actual_annotation_mask_file_path,
            flip_mask=False,
            quantize=False
        )

        actual_annotation_id = actual_annotation_original_file_path.stem[:-9]
        for i in range(1, 6+1):
            assert (
                actual_annotation_id[-i] in "0123456789"
            ), f"ERROR: annotation_ids should end in sixdigits, but this one doesn't: {actual_annotation_id=}."
        
        print(f"Creating {num_fakes_per_actual_annotation} fake annotations from the actual annotation {actual_annotation_id}.")
        for i in range(num_fakes_per_actual_annotation):
            # pick which cutouts to add to the actual annotation, right now all cutouts are added to all actual annotations:  
            
            # choose where to save the fake annotation:
            rid = np.random.randint(0, 1_000_000_000_000_000)
            fake_annotation_id = f"{actual_annotation_id}_fake{rid:015d}"
            fake_original_out_path = fakes_dir / f"{fake_annotation_id}_original.png"
            fake_rgba_out_path = fakes_dir / f"{fake_annotation_id}_nonfloor.png"
            fake_relevance_mask_out_path = fakes_dir / f"{fake_annotation_id}_relevance.png"

            # the relevance mask is boring in that it doesn't matter which cutouts you paste nor where,
            # so you might as well make it right now:
            actual_mask = actual_annotation_rgba_np_u8[:, :, 3]
            PIL.Image.fromarray(255 - actual_mask).save(fake_relevance_mask_out_path)
            
            paste_multiple_cutouts_onto_one_camera_posed_segmentation_annotation(
                cutouts_by_kind=cutouts_by_kind,
                actual_annotation_rgba_np_u8=actual_annotation_rgba_np_u8,  # this is not violated by this procedure.
                camera_pose=camera_pose,  # to get realistics locations and sizes we need to know the camera pose.
                cutout_kind_to_transform=cutout_kind_to_transform, # what albumentations augmentation to use per kind of cutout
                cutout_kind_to_num_cutouts_to_paste=cutout_kind_to_num_cutouts_to_paste,  
                fake_original_out_path=fake_original_out_path,
                fake_rgba_out_path=fake_rgba_out_path,
            )
            prii_verbose = False
            if prii_verbose:
                print("The fake datapoint is these three images:")
                prii(fake_original_out_path, caption=f"{fake_original_out_path}")
                prii(fake_rgba_out_path, caption=f"{fake_rgba_out_path}")
                prii(fake_relevance_mask_out_path, caption=f"{fake_relevance_mask_out_path}")
            fake_counter += 1
            print(f"Finished {fake_counter} out of {how_many_fake_annotations_will_be_made} fake annotations.")
            




if __name__ == "__main__":
    # with launch_ipdb_on_exception():
    shard_id = int(sys.argv[1])
    out_dir = Path(sys.argv[2]).resolve()

    assert 0 <= shard_id < 10, f"ERROR: {shard_id=} should be in 0..9"

    out_dir = Path(sys.argv[2]).resolve()
    make_fake_segmentation_annotations(
        out_dir=out_dir,
        shard_id=shard_id
    )