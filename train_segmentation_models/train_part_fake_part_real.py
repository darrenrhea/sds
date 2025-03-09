from train_a_model import (
     train_a_model
)
from print_yellow import (
     print_yellow
)
import pprint
import sys
from gfposaiwad_get_file_path_of_sha256_assuming_it_was_already_downloaded import (
     gfposaiwad_get_file_path_of_sha256_assuming_it_was_already_downloaded
)
from download_the_files_with_these_sha256s import (
     download_the_files_with_these_sha256s
)
from ltjfthts_load_the_jsonlike_file_that_has_this_sha256 import (
     ltjfthts_load_the_jsonlike_file_that_has_this_sha256
)


def train_part_fake_part_real():
    # this was made by running
    # time python /home/darren/sds/organize_segmentation_data/gather_all_segmentation_annotations.py
    # on lam.

    real_data_sha256 = (
        "6d7074c40a5aa53286f14e8127d2822f9e5ccb68bee112fa6e43f10f4c6a8485"
    )
    
    fake_data_sha256 = (
        "d46eb4099a655c3d3d4c9a98682cdef93e0042e8e433093d6971aa4f11135d1b"
    )

    real_data = ltjfthts_load_the_jsonlike_file_that_has_this_sha256(
        sha256_of_the_jsonlike_file=real_data_sha256,
        check=True
    )
    appropriate_real_data = [
        x
        for x in real_data
        if (
            x["info"]["league"] == "nba"
            and
            "allstar" not in x["clip_id"]
        )
    ]
    pprint.pprint(f"{len(appropriate_real_data)=}")
    
    fake_data = ltjfthts_load_the_jsonlike_file_that_has_this_sha256(
        sha256_of_the_jsonlike_file=fake_data_sha256,
        check=True
    )
    # BEGIN make sure all file assets have been downloaded:
    if True:
        sha256s_to_download = set()
        for x in fake_data:
            sha256s_to_download.add(x["fake_mask_sha256"])
            sha256s_to_download.add(x["fake_original_sha256"])
        for x in appropriate_real_data:
            sha256s_to_download.add(x["mask_sha256"])
            sha256s_to_download.add(x["original_sha256"])
        
        sha256s_to_download = list(sha256s_to_download)
        sha256s_to_download = sorted(sha256s_to_download)

        download_the_files_with_these_sha256s(
            sha256s_to_download=sha256s_to_download,
            max_workers=100,
            verbose=True,
        )
    # ENDOF make sure all file assets have been downloaded.

    fake_data = sorted(
        fake_data,
        key=lambda x: x["fake_original_sha256"]
    )

    how_many_fake_datapoints = int(sys.argv[1])
    fake_data = fake_data[:how_many_fake_datapoints]
    
    num_real_datapoints = len(appropriate_real_data)
    datapoint_path_tuples = []
    for x in fake_data:
        fake_mask_sha256 = x["fake_mask_sha256"]
        fake_original_sha256 = x["fake_original_sha256"]
        mask_path = gfposaiwad_get_file_path_of_sha256_assuming_it_was_already_downloaded(
            sha256=fake_mask_sha256,
            check=False,
        )
        assert mask_path is not None
        original_path = gfposaiwad_get_file_path_of_sha256_assuming_it_was_already_downloaded(
            sha256=fake_original_sha256,
            check=False,
        )
        assert original_path is not None
        datapoint_path_tuples.append(
            (original_path, mask_path, None)
        )
    for x in appropriate_real_data:
        mask_sha256 = x["mask_sha256"]
        original_sha256 = x["original_sha256"]
        mask_path = gfposaiwad_get_file_path_of_sha256_assuming_it_was_already_downloaded(
            sha256=mask_sha256,
            check=False,
        )
        assert mask_path is not None
        original_path = gfposaiwad_get_file_path_of_sha256_assuming_it_was_already_downloaded(
            sha256=original_sha256,
            check=False,
        )
        assert original_path is not None
        datapoint_path_tuples.append(
            (original_path, mask_path, None)
        )

    

    print(f"{datapoint_path_tuples=}")
    num_training_points = len(datapoint_path_tuples)
    print_yellow(f"Between fake and real, that is {num_training_points=}")

    train_a_model(
        datapoint_path_tuples=datapoint_path_tuples,
        other=f"fake{how_many_fake_datapoints}real{num_real_datapoints}",
        resume_checkpoint_path=None,
        drop_a_model_this_often=1,
        num_epochs=100,
    )

if __name__ == "__main__":
    train_part_fake_part_real()