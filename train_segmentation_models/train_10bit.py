from form_checkpoint_prefix import (
     form_checkpoint_prefix
)
from colorama import Fore, Style
import sys
from get_datapoint_path_tuples_from_list_of_dataset_folders import (
     get_datapoint_path_tuples_from_list_of_dataset_folders
)
from train import train
from pathlib import Path


dataset_folders = [
    Path("~/r/bal_game2_bigzoom_floor_10bit").expanduser(),
]

datapoint_path_tuples = get_datapoint_path_tuples_from_list_of_dataset_folders(
    dataset_folders=dataset_folders,
)

for t in datapoint_path_tuples:
    print(t)

print(f"{datapoint_path_tuples=}")
num_training_points = len(datapoint_path_tuples)
print(f"{Fore.YELLOW}{num_training_points=}{Style.RESET_ALL}")


segmentation_convention = "floor"
howmuchdata = f"{len(datapoint_path_tuples)}frames"
patch_width = 1920
patch_height = 1088
model_architecture_family_id = "u3fasternets"
resolution = f"{patch_width}x{patch_height}"
augmentation_strategy_id = "wednesday"
other = "tenbit"
checkpoint_prefix = form_checkpoint_prefix(
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

loss_function_family_id = "mse"
loss_parameters = {}
train_on_binarized_masks = False
num_epochs = 100000
workers_per_gpu = 8
frame_height = 1088
frame_width = 1920
batch_size = 3
do_mixed_precision = True
do_padding = True
train_patches_per_image = 1

resume_checkpoint_path = None
# resume_checkpoint_path = "/shared/checkpoints/u3fasternets-floor-790frames-1920x1088-saveme4_epoch000080.pt"

test_model_interval = 1

resume_optimization = False


train(
    run_id_uuid=None,
    model_architecture_id=model_architecture_family_id,
    augmentation_strategy_id=augmentation_strategy_id,
    checkpoint_prefix=checkpoint_prefix,
    checkpoints_dir=checkpoints_dir,
    loss_function_family_id=loss_function_family_id,
    loss_parameters=loss_parameters,
    train_on_binarized_masks=train_on_binarized_masks,
    num_epochs=num_epochs,
    datapoint_path_tuples=datapoint_path_tuples,
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

