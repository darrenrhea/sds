from typing import Dict
from calculate_model_outputs import calculate_model_outputs
from calculate_loss_function import calculate_loss_function
from loss_info import valid_loss_function_family_ids, validate_loss_parameters
from print_metrics import print_metrics
from model_architecture_info import valid_model_architecture_ids
import time
import copy
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from save_checkpoint import (
     save_checkpoint
)


from colorama import Fore, Style
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

torch.backends.cudnn.benchmark = True


def training_loop_for_multiple_outputs(
    device,
    model_architecture_id: str,
    loss_function_family_id: str,
    loss_parameters: dict,
    model: torch.nn.Module,
    frame_width: int,
    frame_height: int,
    patch_width: int,
    patch_height: int,
    dataloaders: Dict[str, DataLoader],
    optimizer,
    scheduler,
    scaler,
    num_epochs: int,
    checkpoint_prefix: str,
    checkpoints_dir: Path,
    test_model_interval: int,
    WITH_AMP,
):
    """
    This defines the helper function train_model.py,
    which can only be called if you have
    already set up some fairly complicated objects such as
    a model, an optimizer, a scheduler, and a dataloaders for training and validation.
    model_architecture_id
    model,
    optimizer
    scheduler
    num_epochs
    See train.py
    """
    assert (
        isinstance(model, torch.nn.Module)
    ), f"Why is the type of model {type(model)}, not torch.nn.Module?"

    assert isinstance(checkpoints_dir, Path)
    assert (
        checkpoints_dir.is_dir()
    ), f"ERROR: {checkpoints_dir} is not an extant directory"

    assert model_architecture_id in valid_model_architecture_ids
    assert loss_function_family_id in valid_loss_function_family_ids
    validate_loss_parameters(
        loss_function_family_id=loss_function_family_id,
        loss_parameters=loss_parameters
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        since = time.time()

        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #for param_group in optimizer.param_groups:
                #    print("LR", param_group['lr'])

                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            # we should not validate every epoch because it takes time:
            if not (epoch % test_model_interval == 0 or epoch == num_epochs) and phase == 'val':
                print(f"{Fore.RED}I am skipping validation because {epoch=}{Style.RESET_ALL}")
                continue
            if phase == 'train':
                for batch_of_channel_stacks in tqdm(dataloaders[phase]):
                    inputs = batch_of_channel_stacks[:, :3, :, :]
                    labels = batch_of_channel_stacks[:, 3:5, :, :]
                    importance_weights = None
                    assert labels.ndim == 4, "even if there has to be a trivial channel dimension, the labels should be 4D, not 3D"
                    # assert importance_weights.size() == labels.size(), "the importance weights should be the same size as the labels"
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    # importance_weights = importance_weights.cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        with WITH_AMP():
                            dict_of_output_tensors = calculate_model_outputs(
                                model=model,
                                model_architecture_id=model_architecture_id,
                                inputs=inputs,
                                train=True
                            )
                            
                            # this call mutates the metrics:
                            loss = calculate_loss_function(
                                loss_function_family_id=loss_function_family_id,
                                loss_parameters=loss_parameters,
                                dict_of_output_tensors=dict_of_output_tensors,
                                labels=labels,
                                importance_weights=importance_weights,
                                metrics=metrics
                            )

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                # loss.backward()
                                scaler.scale(loss).backward()
                                #optimizer.step()
                                scaler.step(optimizer)
                                scaler.update()
                            

                    # statistics
                    epoch_samples += inputs.size(0)

                print_metrics(metrics, epoch_samples, phase)
                epoch_loss = metrics['loss'] / epoch_samples
                tqdm.write(f'epoch loss {epoch_loss:g}')

            # deep copy the model
            if phase == 'val':
                checkpoint_extra = {
                    'model_architecture_id': model,
                    'frame_width': frame_width,
                    'frame_height': frame_height,
                    'patch_width': patch_width,
                    'patch_height': patch_height,
                }
                # TODO: wire thru save checkpoint frequency
                if epoch % test_model_interval == 0 or epoch == num_epochs:   # save periodically and also one last time.
                    out_path = checkpoints_dir / f"{checkpoint_prefix}_epoch{epoch:06d}.pt"
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        out_path=out_path,
                        scheduler=scheduler,
                        extra = checkpoint_extra)
                    tqdm.write(f'saved checkpoint {out_path}')



                # prepare next training epoch
                scheduler.step()

              

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('best val loss: {:g}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
