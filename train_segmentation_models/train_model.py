from insert_progress_message import (
     insert_progress_message
)
import uuid
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
import pprint


from colorama import Fore, Style
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

torch.backends.cudnn.benchmark = True


def train_model(
    run_id_uuid: uuid.UUID,
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
    time_training_began = time.time()
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

    num_samples_trained = 0
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        

        # BEGIN training an epoch:
        metrics = defaultdict(float)
        epoch_samples = 0
        model.train()
        for batch_of_channel_stacks in tqdm(dataloaders["train"]):
            inputs = batch_of_channel_stacks[:, :3, :, :]
            labels = batch_of_channel_stacks[:, 3:4, :, :]
            importance_weights = batch_of_channel_stacks[:, 4:5, :, :]
            assert labels.ndim == 4, "even if there has to be a trivial channel dimension, the labels should be 4D, not 3D"
            assert importance_weights.size() == labels.size(), "the importance weights should be the same size as the labels"
            inputs = inputs.cuda()
            labels = labels.cuda()
            importance_weights = importance_weights.cuda()


            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
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

                    # loss.backward()
                    scaler.scale(loss).backward()
                    #optimizer.step()
                    scaler.step(optimizer)
                    scaler.update()
                    

            # statistics
            epoch_samples += inputs.size(0)
        
        scheduler.step()
        
        print_metrics(metrics, epoch_samples, phase="train")
        print(f"train {epoch_samples=}")
        num_samples_trained += epoch_samples
        # ENDOF training an epoch.


        # we should not validate every epoch because it takes time:
        if epoch % test_model_interval == 0 or epoch == num_epochs or epoch == 1:
            do_validation = True
        else:
            print(f"{Fore.RED}I am skipping the validation phase because {epoch=}{Style.RESET_ALL}")
            do_validation = False
        
        if do_validation:
            # BEGIN validation:
            metrics = defaultdict(float)
            epoch_samples = 0
            model.eval()
            for batch_of_channel_stacks in tqdm(dataloaders['val']):
                inputs = batch_of_channel_stacks[:, :3, :, :]
                labels = batch_of_channel_stacks[:, 3:4, :, :]
                importance_weights = batch_of_channel_stacks[:, 4:5, :, :]
                assert labels.ndim == 4, "even if there has to be a trivial channel dimension, the labels should be 4D, not 3D"
                assert importance_weights.size() == labels.size(), "the importance weights should be the same size as the labels"
                inputs = inputs.cuda()
                labels = labels.cuda()
                importance_weights = importance_weights.cuda()

                with torch.set_grad_enabled(False):
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

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase="val")
            l1_loss = metrics["mse"] / epoch_samples
            final_metrics = dict(
                l1_loss=l1_loss,
            )
            pprint.pprint(final_metrics)
            time_elapsed = time.time() - time_training_began
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


            progress_message_jsonable = dict(
                seconds_elapsed=time_elapsed,
                num_samples_trained=num_samples_trained,
                metrics=final_metrics,
            )
            if run_id_uuid is not None:
                insert_progress_message(
                    run_id_uuid=run_id_uuid,
                    progress_message_jsonable=progress_message_jsonable,
                )
            # ENDOF validation.

        
            checkpoint_extra = {
                'model_architecture_id': model,
                'frame_width': frame_width,
                'frame_height': frame_height,
                'patch_width': patch_width,
                'patch_height': patch_height,
            }
            # TODO: wire thru save checkpoint frequency            
            
            out_path = checkpoints_dir / f"{checkpoint_prefix}_epoch{epoch:06d}.pt"
            
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                out_path=out_path,
                scheduler=scheduler,
                extra = checkpoint_extra)
            tqdm.write(f'saved checkpoint {out_path}')



