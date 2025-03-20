from get_datapoint_path_tuples_for_bal import (
     get_datapoint_path_tuples_for_bal
)
from insert_run_id import (
     insert_run_id
)
from pathlib import Path
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
from print_green import (
     print_green
)


def train_part_fake_part_real():
    # this was made by running
    # time python /home/darren/sds/organize_segmentation_data/gather_all_segmentation_annotations.py
    # on lam.

    real_data_sha256 = (
        "6d7074c40a5aa53286f14e8127d2822f9e5ccb68bee112fa6e43f10f4c6a8485"  # this is 1541 real annotations, 1425 of which are nba
    )
    
    # python organize_segmentation_data/gather_all_fake_segmentation_annotations.py
    fake_data_sha256 = (
        # "d46eb4099a655c3d3d4c9a98682cdef93e0042e8e433093d6971aa4f11135d1b"  # allstars 5408 of them
        # "4a04e0dc13f03d9cb6956060044740f98710c8c1d9c19ec8dde8da27939b1e44" # BAL, albeit only partially generated, 877
        "4d05435f93db1d0bfaf431c8df27edc61d88b9c21a1819f61d4af9fa64c3d8db" # fully generated BAL, 2704 
    )

    run_description_jsonable = dict(
        real_data_sha256=real_data_sha256,
        fake_data_sha256=fake_data_sha256,
    )

    run_id_uuid = insert_run_id(
        run_description_jsonable=run_description_jsonable
    )
    
    print_green(
        f"run_id = {str(run_id_uuid)}"
    )

    # resume_checkpoint_path = None
    # resume_checkpoint_path = get_file_path_of_sha256("32b6b27bc294dee980b62df7dd7950f975781e3c78e6223af4b1f8ea41cbb309")
    # i.e. u3fasternets-floor-10911frames-1920x1088-citydec27_epoch000001.pt"
    # resume_checkpoint_path = Path("/shared/checkpoints/u3fasternets-floor-2302frames-1920x1088-fake877real1425_epoch000559.pt")
    # resume_checkpoint_path = Path("/shared/checkpoints/u3fasternets-floor-7914frames-1920x1088-fake2704real1425_epoch000006.pt")
    resume_checkpoint_path = Path("/shared/checkpoints/u3fasternets-floor-9114frames-1920x1088-fake2704real1425_epoch000076.pt")
    

    if resume_checkpoint_path is not None:
        assert resume_checkpoint_path.is_file(), f"{resume_checkpoint_path} is not an extant file"


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
    num_real_datapoints = len(appropriate_real_data)
    print_yellow(f"Using {num_real_datapoints=}")
    
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
    how_many_fake_datapoints = min(how_many_fake_datapoints, len(fake_data))
    print_yellow(f"Using {how_many_fake_datapoints=} (MAX possible is {len(fake_data)})")
    fake_data = fake_data[:how_many_fake_datapoints]
    
    fake_datapoint_path_tuples = []
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
        fake_datapoint_path_tuples.append(
            (original_path, mask_path, None)
        )

    analogous_datapoint_path_tuples = []
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
        analogous_datapoint_path_tuples.append(
            (original_path, mask_path, None)
        )

    fixups = get_datapoint_path_tuples_for_bal()
    upmultiplied_analogous_datapoint_path_tuples = analogous_datapoint_path_tuples * 2
    upmultiplied_fixups = []
    for _ in range(40):
        upmultiplied_fixups += fixups
    
    print(f"{len(fake_datapoint_path_tuples)=}")
    print(f"{len(upmultiplied_analogous_datapoint_path_tuples)=}")
    print(f"{len(upmultiplied_fixups)=}")

    datapoint_path_tuples = (
        fake_datapoint_path_tuples
        +
        upmultiplied_analogous_datapoint_path_tuples
        +
        upmultiplied_fixups
    )

    # print(f"{datapoint_path_tuples=}")
    num_training_points = len(datapoint_path_tuples)
    print_yellow(f"Between fake and real, that is {num_training_points=}")

    train_a_model(
        run_id_uuid=run_id_uuid,
        datapoint_path_tuples=datapoint_path_tuples,
        other=f"fake{how_many_fake_datapoints}real{num_real_datapoints}",
        resume_checkpoint_path=resume_checkpoint_path,
        drop_a_model_this_often=1,
        num_epochs=100000,
    )


if __name__ == "__main__":
    train_part_fake_part_real()