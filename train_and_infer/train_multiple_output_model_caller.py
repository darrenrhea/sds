import json
from train_multiple_output_model import (
     train_multiple_output_model
)
import pprint
from get_local_file_paths_for_annotations import (
     get_local_file_paths_for_annotations
)
from form_checkpoint_prefix import (
     form_checkpoint_prefix
)
from pathlib import Path

# There is a relatively small file that describes all the metadata about ALL annotations:
video_frame_annotations_metadata_sha256 = (
    "4bffcd3e6d1e6cdc0055fdf5004b498e1f07282ddeebe2d524a59d28726208d2"
)

desired_labels = set(["depth_map", "floor_not_floor", "original"])

local_file_pathed_annotations = get_local_file_paths_for_annotations(
    video_frame_annotations_metadata_sha256=video_frame_annotations_metadata_sha256,
    desired_labels=desired_labels,
    # desired_clip_id_frame_index_pairs=[
    #     ('SL_2022_00', 133500),
    #     ('SL_2022_00', 136100),
    #     ('CHAvNYK_PGM_city_bal_12-09-2022', 267000),
    # ],
    max_num_annotations=None,
    print_in_iterm2=False,
)


for a in local_file_pathed_annotations:
    # print("")
    # pprint.pprint(local_file_pathed_annotations)
    assert a["local_file_paths"]["original"].is_file()
    assert a["local_file_paths"]["floor_not_floor"].is_file()
    assert a["local_file_paths"]["depth_map"].is_file()
    


num_training_points = len(local_file_pathed_annotations)
segmentation_convention = "floor"
howmuchdata = f"{len(local_file_pathed_annotations)}frames"
patch_width = 1920
patch_height = 1088
model_architecture_family_id = "u3fasternets"
resolution = f"{patch_width}x{patch_height}"
augmentation_strategy_id = "wednesday" # felix3" # imagemaskmotionblur
other = "floornotfloordepthmap"

# This is silly.  We are trying to write all the metadata into a file name:
checkpoint_prefix = form_checkpoint_prefix(
    other=other,
    segmentation_convention=segmentation_convention,
    num_training_points=num_training_points,
    patch_height=patch_height,
    patch_width=patch_width,
    model_architecture_family_id=model_architecture_family_id,
)

model_metadata = dict(
    other=other,
    segmentation_convention=segmentation_convention,
    num_training_points=num_training_points,
    patch_height=patch_height,
    patch_width=patch_width,
    model_architecture_family_id=model_architecture_family_id,
)

checkpoints_dir = Path(
    "/shared/checkpoints"
)

loss_function_family_id = "unweighted_l1"
loss_parameters = {}
num_epochs = 100000
workers_per_gpu = 1
frame_height = 1088
frame_width = 1920
batch_size = 3
do_mixed_precision = True
do_padding = True
train_patches_per_image = 1

resume_checkpoint_path = None
# resume_checkpoint_path = "/shared/checkpoints/u3fasternets-floor-1615frames-1920x1088-wednesday-nba2024finalgame4_epoch000036.pt"
# resume_checkpoint_path = "/shared/checkpoints/u3fasternets-floor-1565frames-1920x1088-wednesday-nba2024finalgame4_epoch000015.pt"
# resume_checkpoint_path = "/shared/checkpoints/u3fasternets-floor-1271frames-1920x1088-wednesday-nba2024finalgame4_epoch000098.pt"
# resume_checkpoint_path = "/shared/checkpoints/u3fasternets-floor-1265frames-1920x1088-wednesday_epoch000134.pt"
# resume_checkpoint_path = "/shared/checkpoints/u3fasternets-floor-875frames-1920x1088-wednesday-resumed-bay_mta_epoch000875.pt"
test_model_interval = 1

resume_optimization = False

# Write the metadata that will be saved alongside as a sister of the model:
resume_checkpoint_path_str = str(resume_checkpoint_path) if resume_checkpoint_path else None

model_training_experience_metadata = dict(
    other=other,
    segmentation_convention=segmentation_convention,
    num_training_points=num_training_points,
    patch_height=patch_height,
    patch_width=patch_width,
    model_architecture_family_id=model_architecture_family_id,
    batch_size=batch_size,
    frame_height=frame_height,
    frame_width=frame_width,
    do_mixed_precision=do_mixed_precision,
    do_padding=do_padding,
    train_patches_per_image=train_patches_per_image,
    workers_per_gpu=workers_per_gpu,
    loss_function_family_id=loss_function_family_id,
    loss_parameters=loss_parameters,
    resume_checkpoint_path_str=resume_checkpoint_path_str,
)

# make sure it is jsonable so that it does not crash later:
json.dumps(model_metadata, indent=4)


train_multiple_output_model(
    model_training_experience_metadata=model_training_experience_metadata,
    model_architecture_id=model_architecture_family_id,
    augmentation_strategy_id=augmentation_strategy_id,
    checkpoint_prefix=checkpoint_prefix,
    checkpoints_dir=checkpoints_dir,
    loss_function_family_id=loss_function_family_id,
    loss_parameters=loss_parameters,
    num_epochs=num_epochs,
    local_file_pathed_annotations=local_file_pathed_annotations,
    workers_per_gpu=workers_per_gpu,
    frame_height=frame_height,
    frame_width=frame_width,
    patch_width=patch_width,
    patch_height=patch_height,
    batch_size=batch_size,
    do_mixed_precision=do_mixed_precision,
    do_padding=do_padding,
    train_patches_per_image=train_patches_per_image,
    resume_checkpoint_path=resume_checkpoint_path,
    test_model_interval=test_model_interval,
    resume_optimization=resume_optimization
)

resulting_checkpoint = checkpoints_dir / f"{checkpoint_prefix}_epoch{num_epochs}.pth"

