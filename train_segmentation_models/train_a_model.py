import uuid
from form_checkpoint_prefix import (
     form_checkpoint_prefix
)
from colorama import Fore, Style
from typing import List, Tuple, Optional
from train import train
from pathlib import Path


def train_a_model(
    run_id_uuid: uuid.UUID,
    datapoint_path_tuples: List[Tuple[Path, Path, Optional[Path]]],
    other: str,
    resume_checkpoint_path: Optional[str],
    drop_a_model_this_often: int,
    num_epochs: Optional[int],
):

    num_training_points = len(datapoint_path_tuples)
    print(f"{Fore.YELLOW}{num_training_points=}{Style.RESET_ALL}")

    segmentation_convention = "floor"
    patch_width = 1920
    patch_height = 1088
    model_architecture_family_id = "u3fasternets"
    augmentation_strategy_id = "wednesday" # felix3" # imagemaskmotionblur

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
    if num_epochs is None:
        num_epochs = 100000
    
    workers_per_gpu = 8
    frame_height = 1088
    frame_width = 1920
    batch_size = 8
    do_mixed_precision = True
    do_padding = True
    train_patches_per_image = 1
    test_model_interval = drop_a_model_this_often

    resume_optimization = False


    train(
        run_id_uuid=run_id_uuid,
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

