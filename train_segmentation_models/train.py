from get_normalization_and_chw_transform import (
     get_normalization_and_chw_transform
)
from DummyWith import (
     DummyWith
)
import numpy as np
from adan import Adan
from blackpad_preprocessor import blackpad_preprocessor
from load_datapoints_in_parallel import load_datapoints_in_parallel
from typing import Optional, List
from train_model import train_model
from model_architecture_info import valid_model_architecture_ids
from get_training_augmentation import get_training_augmentation
from pathlib import Path
from loss_info import valid_loss_function_family_ids

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.optim import lr_scheduler


from unettools import MODEL_LOADERS
from load_checkpoint import load_checkpoint
from WarpDataset import WarpDataset

from colorama import Fore, Style
import ssl
# from visualize_dataset import visualize_dataset
from typing import Tuple

import torchvision.models as models

torch.backends.cudnn.benchmark = True
ssl._create_default_https_context = ssl._create_unverified_context


def train(
    model_architecture_id: str,
    augmentation_strategy_id: str,
    checkpoint_prefix: str,
    checkpoints_dir: Path,
    loss_function_family_id: str,
    loss_parameters: dict,
    train_on_binarized_masks: bool,
    num_epochs: int,
    datapoint_path_tuples: List[Tuple[Path, Path, Optional[Path]]],
    workers_per_gpu: int,
    frame_height: int,
    frame_width: int,
    patch_width: int,
    patch_height: int,
    batch_size: int,
    do_mixed_precision: bool,
    do_padding: bool,
    train_patches_per_image: int,
    resume_checkpoint_path: Optional[Path],
    test_model_interval: int,
    resume_optimization: bool = False,  # be careful, you might not know what this is
):
    """
    This Python procedure trains a segmentation model.
    It is not a CLI, but rather a Python function. However,
    if you want to call it from the command line,
    see train_cli.py which raps it as a executable script
    callable from the command line.
    """
    print(
f"""{Fore.YELLOW}
Doing this:

train(
    model_architecture_id={model_architecture_id},
    augmentation_strategy_id={augmentation_strategy_id},
    checkpoint_prefix={checkpoint_prefix},
    checkpoints_dir={checkpoints_dir},
    loss_function_family_id={loss_function_family_id},
    loss_parameters={loss_parameters},
    train_on_binarized_masks={train_on_binarized_masks},
    num_epochs={num_epochs},
    datapoint_path_tuples=datapoint_path_tuples,
    workers_per_gpu={workers_per_gpu},
    frame_height={frame_height},
    frame_width={frame_width},
    patch_width={patch_width},
    patch_height={patch_height},
    batch_size={batch_size},
    do_mixed_precision={do_mixed_precision},
    do_padding={do_padding},
    train_patches_per_image={train_patches_per_image},
    resume_checkpoint_path={resume_checkpoint_path},
    test_model_interval={test_model_interval},
    resume_optimization={resume_optimization}
)
{Style.RESET_ALL}
"""
)
    
    assert (
        isinstance(checkpoints_dir, Path)
    ), f"ERROR: {checkpoints_dir} is not a Path, maybe you gave something of type {type(checkpoints_dir)=} namely {checkpoints_dir=}"

    assert (
        checkpoints_dir.is_dir()
    ), f"ERROR: {checkpoints_dir} is not an extant directory"
    
    assert model_architecture_id in valid_model_architecture_ids, f"unknown model architecture: {model_architecture_id}"

    assert loss_function_family_id in valid_loss_function_family_ids, f"unknown loss: {loss_function_family_id}"
    assert isinstance(num_epochs, int), f"num_epochs must be an integer, not {type(num_epochs)}"
    assert num_epochs > 0, f"num_epochs must be positive, not {num_epochs}"    
    assert isinstance(workers_per_gpu, int), f"workers_per_gpu must be an integer, not {type(workers_per_gpu)}"
    assert workers_per_gpu > 0, f"workers_per_gpu must be positive, not {workers_per_gpu}"


    print(f"{Fore.YELLOW}{frame_height=}, {frame_width=}{Style.RESET_ALL}")


    WITH_AMP = torch.cuda.amp.autocast if do_mixed_precision else DummyWith

    if frame_width == patch_width and frame_height == patch_height:
        assert train_patches_per_image == 1, "There is no point in training with more than one patch per image if the patch size is the same as the frame size."
   
    scaler = torch.cuda.amp.GradScaler()

    # always 2 classes
    if loss_function_family_id in ["awl", "mse"]:
        num_class = 1
    else:
        num_class = 2

    in_channels = 3


    print(f"{in_channels=}, {num_class=}")


    if model_architecture_id not in MODEL_LOADERS:
        raise Exception(f'unknown model: {model_architecture_id}')
    
    assert num_class == 1
    
    model = MODEL_LOADERS[model_architecture_id](
        in_channels = in_channels,
        num_class = num_class
    )

    if resume_checkpoint_path:
        checkpoint = load_checkpoint(
            in_path = resume_checkpoint_path,
            model = model,
            multigpu = True
        )

    # TODO: somehow determine what normalization to use
    normalization_and_chw_transform = get_normalization_and_chw_transform()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('running on', device)


    # model = ResNet()
    print(f"{type(model)=}")
    assert isinstance(model, nn.Module)

    if torch.cuda.device_count() > 1:
        print("Using multiple GPUs because they are available")
        print(f"using {torch.cuda.device_count()=} gpus")
        # this isnt recommended, see
        # https://pytorch.org/docs/stable/notes/cuda.html#cuda-nn-ddp-instead
        model = nn.DataParallel(model).cuda()
        x = torch.randn(8, 3, patch_height, patch_width).cuda()
        out = model(x)

    else:
        print("Only going to use one GPU for training")
        model = model.cuda()

    
    augment = get_training_augmentation(
        augmentation_id=augmentation_strategy_id, 
        frame_width=frame_width, 
        frame_height=frame_height
    )
    assert augment is not None, f"ERROR: unknown augmentation strategy: {augmentation_strategy_id}"
   
    # load dataset
    
   
    if do_padding:
        assert frame_height == 1088, "doing padding to force frames to 1920x1088"
        assert frame_width == 1920, "doing padding to force frames to 1920x1088"
        print(f"{Fore.YELLOW}WARNING: doing padding to force frames to 1920x1088{Style.RESET_ALL}")

        # preprocessor = reflect_preprocessor  # fix this
        preprocessor = blackpad_preprocessor

        preprocessor_params = dict(
            desired_height=frame_height,
            desired_width=frame_width
        )
    else:
        preprocessor = None
        preprocessor_params = dict()

    channel_stacks = load_datapoints_in_parallel(
        datapoint_path_tuples = datapoint_path_tuples,
        preprocessor = preprocessor,
        preprocessor_params = preprocessor_params,
    )

    for channel_stack in channel_stacks:
        assert channel_stack.shape[0] == frame_height, f"{channel_stack.shape=} {frame_height=}"
        assert channel_stack.shape[1] == frame_width
        assert channel_stack.shape[2] == 5, "aren't we expecting 5 channels since we are using relevance masks?"
        assert channel_stack.dtype == np.uint8
        assert channel_stack.ndim == 3
        assert np.sum(channel_stack[:, :, 4].astype(np.float32)) > 0.0, "there should be some onscreen/relevant part"
    


    print(f"{type(augment)=}")
    train_set = WarpDataset(
        channel_stacks = channel_stacks,
        train_patches_per_image = train_patches_per_image,
        patch_width = patch_width,
        patch_height = patch_height,
        output_binarized_masks = train_on_binarized_masks,
        normalization_and_chw_transform = normalization_and_chw_transform,
        augment = augment,
        num_mask_channels = num_class
    )

    val_set = WarpDataset(
        channel_stacks = channel_stacks,
        train_patches_per_image = train_patches_per_image,
        patch_width = patch_width,
        patch_height = patch_height,
        output_binarized_masks = train_on_binarized_masks,
        normalization_and_chw_transform = normalization_and_chw_transform,
        augment = augment,
        num_mask_channels = num_class
    )

    
    # visualize_dataset(data_set=train_set, num_samples=5)


    dataloaders = {
        'train': DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers_per_gpu * torch.cuda.device_count()
        ),
        'val': DataLoader(
            dataset=val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers_per_gpu * torch.cuda.device_count()
        ),
    }


    # Prove that the dataloaders are iterables such that if you
    # turn them into and iterator (like for looping over them)
    # with each iteration you get one torch.Tenosr of shape
    # batch_size x 5 channels x 1088 height x 1920 width
    thing = next(iter(dataloaders['train']))
    assert isinstance(thing, torch.Tensor)
    assert thing.device.type == 'cpu'
    assert thing.shape == (batch_size, 5, patch_height, patch_width) 

    
    optimizer_ft = Adan(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=0.02,
    )
    

    scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=500, gamma=0.5)


    if resume_optimization:
        if 'optimizer' in checkpoint:
            optimizer_ft.load_state_dict(checkpoint['optimizer'])
        if 'lr_scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['lr_scheduler'])

    model = train_model(
        device = device,
        model_architecture_id = model_architecture_id,
        loss_function_family_id = loss_function_family_id,
        loss_parameters = loss_parameters,
        model = model,
        frame_width = frame_width,
        frame_height = frame_height,
        patch_width = patch_width,
        patch_height = patch_height,
        dataloaders = dataloaders,
        optimizer = optimizer_ft,
        scheduler = scheduler,
        scaler = scaler,
        num_epochs = num_epochs,
        checkpoint_prefix = checkpoint_prefix, 
        checkpoints_dir = checkpoints_dir,
        test_model_interval = test_model_interval,
        WITH_AMP = WITH_AMP
    )


