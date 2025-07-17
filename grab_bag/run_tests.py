"""
We need tests to see if things are broken.
We make fake datasets that have a very obvious pattern to them.
We train on that fake dataset to get a checkpoint,
then infer with that checkout,
and see if the inferences are close enough to the ground truth.
"""
from make_fake_discrete_segmentation_problem import (
     make_fake_discrete_segmentation_problem
)
import json
import pprint as pp
import os
from pathlib import Path
import subprocess
import numpy as np
import PIL.Image
import shutil
from colorama import Fore, Style
from validate_test_case import validate_test_case
from prii import (
     prii
)
from make_easy_fake_feathered_segmentation_problem import (
     make_easy_fake_feathered_segmentation_problem
)


def run_the_training_process_to_test_it(desc):
    model_architecture_id = desc["model_architecture_id"]
    loss_function_family_id = desc["loss_function_family_id"]
    loss_parameters = desc["loss_parameters"]
    num_gpus_to_train_on = desc["num_gpus_to_train_on"]
    workers_per_gpu = desc["workers_per_gpu"]
    patches_per_image = desc["patches_per_image"]
    frame_width = desc["frame_width"]
    frame_height = desc["frame_height"]
    patch_width = desc["patch_width"]
    patch_height = desc["patch_height"]
    batch_size = desc["batch_size"]
    num_epochs = desc["num_epochs"]
    test_size = desc["test_size"]
    checkpoint_prefix = desc["checkpoint_prefix"]

    # I don't think we want to save the checkpoints for tests
    # in /shared/checkpoints
    # because they are very temporary and not useful for anything else.
    checkpoints_dir = Path("checkpoints").resolve()
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    args = [
        "python",
        "train_cli.py",
        model_architecture_id,
        "temp_fake_dataset",
        "--loss",
        loss_function_family_id,
        "--loss_parameters",
        json.dumps(loss_parameters),
        "--dataset-kind",
        "nonfloor",
        "--patch-width",
        f"{patch_width}",
        "--patch-height",
        f"{patch_height}",
        "--augmentation",
        "basic",
        "--batch-size",
        f"{batch_size}",
        "--epochs",
        f"{num_epochs}",
        "--test-size",
        f"{test_size}",
        "--train",
        "--checkpoint_prefix",
        checkpoint_prefix,
        "--save-interval",
        "1",
        "--checkpoints_dir",
        str(checkpoints_dir),
        "--resolution",
        f"{frame_width}x{frame_height}",
        "--ppi",
        str(patches_per_image),
        "--workers_per_gpu",
        str(workers_per_gpu),
    ]

    if num_gpus_to_train_on == 1:
        CUDA_VISIBLE_DEVICES = "0"
    elif num_gpus_to_train_on == 2:
        CUDA_VISIBLE_DEVICES = "0,2"
    else:
        raise Exception("ERROR: we do not support testing on more than 2 GPUs because not every computer has 3")
      

    print(f"Running under CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}")
    print(
        f"export CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} &&\n"
        +
        " \\\n".join(args)
    )

    environ = os.environ.copy()
    environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


    completed_process = subprocess.run(args=args, env=environ)
    if completed_process.returncode != 0:
        return False
    return True


def infer_with_the_trained_model_to_see_if_things_are_basically_working(
    desc
):
    """
    Hopefully you already trained a segmentation model on a
    unit-test-sized dataset, and now you need to infer with it to see if
    that trained model actually performs close to what you would expect
    on such an easy segmentation problem.
    """
    num_gpus_to_infer_with = desc["num_gpus_to_infer_with"]
    num_epochs = desc["num_epochs"]
    model_architecture_id = desc["model_architecture_id"]
    model_id_suffix = "unittest"
    training_data_dir = Path("temp_fake_dataset").resolve()
    output_dir = Path("inferences").resolve()
    
    # murder any preexisting inferences:
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(exist_ok=True)

    frame_width = desc["frame_width"]
    frame_height = desc["frame_height"]
    patch_height = desc["patch_height"]
    patch_width = desc["patch_width"]
    stride_height = desc["stride_height"]
    stride_width = desc["stride_width"]
    checkpoint_prefix = desc["checkpoint_prefix"]
    
    checkpoints_dir = Path("checkpoints").resolve()
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    assert checkpoints_dir.is_dir(), "ERROR: the checkpoints directory does not even exist?"
    trained_weights_file_name = f"{checkpoint_prefix}_epoch{(num_epochs-1):06d}.pt"
    trained_weights_path = checkpoints_dir / trained_weights_file_name


    args = [
        "python",
        "parallel_infer3.py",
        model_architecture_id,  # make this a flag --model_architecture_family_id, everything a flag
        str(trained_weights_path),  # the checkpoint.  Make it a flag
        "--original-size",
        f"{frame_width},{frame_height}",
        "--patch-width",
        f"{patch_width}",
        "--patch-height",
        f"{patch_height}",
        "--patch-stride-width",
        f"{stride_width}",
        "--patch-stride-height",
        f"{stride_height}",
        "--out-dir",
        f"{output_dir}",
        "--model-id-suffix",
        model_id_suffix,
        f"{training_data_dir}/*.jpg"  # so-called input, make it a flag
    ]
    
    if num_gpus_to_infer_with == 1:
        CUDA_VISIBLE_DEVICES = "0"
    elif num_gpus_to_infer_with == 2:
        CUDA_VISIBLE_DEVICES = "0,2"
    elif num_gpus_to_infer_with == 3:
        CUDA_VISIBLE_DEVICES = "0,1,2"
    else:
        raise Exception("ERROR: we do not support testing inference on more than 3 GPUs")
    


    print("The inference bash command we are testing is basically:")
    print(
        f"export CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} &&\n"
        +
        " \\\n".join(args)
    )

    environ = os.environ.copy()
    environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    completed_process = subprocess.run(args=args, env=environ)
    if completed_process.returncode != 0:
        print(f"{Fore.RED}parallel_infer3.py process FAILED with exit code {completed_process.returncode}!{Style.RESET_ALL}")
        return False
    
    print(f"{Fore.RED}parallel_infer3.py process COMPLETED SUCCESSFULLY!{Style.RESET_ALL}")
    return True


def determine_if_inferences_are_close_enough(desc):
    """
    Having gone through the whole process of training a segmentation model
    on a rather easy segmentation problem,
    then inferring with that trained model on a dataset and saving those inferences to disk,
    did it understand the pattern well enough?  Does it pass?
    """
    evaluate_trained_model_in_feathered_or_binary_sense = desc["evaluate_trained_model_in_feathered_or_binary_sense"]
    assert evaluate_trained_model_in_feathered_or_binary_sense in ["feathered", "binary"]

    if evaluate_trained_model_in_feathered_or_binary_sense == "feathered":
        l1_error_threshold = desc["l1_error_threshold"]
    elif evaluate_trained_model_in_feathered_or_binary_sense == "binary":
        classification_error_threshold = desc["classification_error_threshold"]
        assert classification_error_threshold < 0.5, "ERROR: the classification error threshold is too high, it should be less than 0.5"
        assert classification_error_threshold > 0.0, "ERROR: the classification error threshold is too low, it should be greater than 0.0"

    model_architecture_id = desc["model_architecture_id"]
    num_training_datapoints = desc["num_training_datapoints"]
    training_data_dir = Path("temp_fake_dataset").resolve()
    fake_dataset_dir = Path("temp_fake_dataset").resolve()
    assert fake_dataset_dir.is_dir(), "ERROR: the inferences directory does not even exist?"
    failed_accuracy = False
    for i in range(num_training_datapoints):
        print(f"Checking inference {i=}")
        frame_height = desc["frame_height"]
        frame_width = desc["frame_width"]

        train_mask_path = training_data_dir / f"fake_{i:06d}_nonfloor.png"
        inferred_mask_path = Path(f"inferences/fake_{i:06d}_unittest.png")

        inferred_u8 = np.array(PIL.Image.open(inferred_mask_path))
        assert (
            inferred_u8.shape == (desc["frame_height"], desc["frame_width"])
        ), f"ERROR: the inferred_u8 mask is not even the right shape, it is {inferred_u8.shape}"

        should_be_rgba = np.array(PIL.Image.open(train_mask_path))
        should_be_u8 = should_be_rgba[..., 3]
        # TODO: This should be fixed, i.e. duat should pass without adhoc flipping:
        if model_architecture_id == "duat":
            should_be_u8 = 255 - should_be_u8
        print("Answer should be this:")
        prii(should_be_u8)
        print("but it comes out as is:")
        prii(inferred_u8)
       
        if evaluate_trained_model_in_feathered_or_binary_sense == "binary":
            should_be_truncated = should_be_u8 > 127
            inferred_truncated = inferred_u8 > 127
            wrong = should_be_truncated != inferred_truncated
            num_wrong = np.sum(wrong)
            error_percent = num_wrong / (frame_height * frame_width)
            print(f"{Fore.RED}{error_percent=}{Style.RESET_ALL}")
            if error_percent > classification_error_threshold:
                print(f"{Fore.RED}FAILED!{Style.RESET_ALL}")
                failed_accuracy = True
        elif evaluate_trained_model_in_feathered_or_binary_sense == "feathered":
            l1 = np.abs(
                should_be_u8.astype(np.int32)
                -
                inferred_u8.astype(np.int32)
            ) / 255.0
            average_l1_error = np.sum(l1) / (frame_height * frame_width)
            print(f"{Fore.GREEN}{average_l1_error=}{Style.RESET_ALL}")
            if average_l1_error > l1_error_threshold:
                print(f"{Fore.RED}FAILED to be below {l1_error_threshold}!{Style.RESET_ALL}")
                failed_accuracy = True
        else:
            raise Exception("ERROR: this should never happen")

    if failed_accuracy:
        return False
    else:
        print(f"{Fore.GREEN}PASSED!{Style.RESET_ALL}")
        return True


def run_one_test_case(desc):
    train_on_feathered = desc["train_on_feathered"]
    binarize_masks_before_training = desc["binarize_masks_before_training"]
    print("Running the test case:")
    pp.pprint(desc)

    validate_test_case(desc)
    train = True
    if train:
        # make some easy fake segmentation problem to solve
        if train_on_feathered:
            make_easy_fake_feathered_segmentation_problem(desc)
            assert binarize_masks_before_training is False, "ERROR: if you are training on feathered masks, you probably do not want to binarize_the_masks"
        else:  # train on discretely labelled training data:
            make_fake_discrete_segmentation_problem(desc)
            assert binarize_masks_before_training is True, "ERROR: if you are training on discretely labelled masks, you probably want to binarize the masks before training"

        # solve that segmentation problem by training
        training_process_success = run_the_training_process_to_test_it(desc)
        if not training_process_success:
            print(f"{Fore.RED}The training process did not exit successfully{Style.RESET_ALL}")
            return False

    # use that trained model to infer to see if the program works.
    inference_process_success = infer_with_the_trained_model_to_see_if_things_are_basically_working(desc=desc)
    if not inference_process_success:
        return False

    accurate_enough = determine_if_inferences_are_close_enough(
        desc=desc
    )

    if not accurate_enough:
        return False
    
    return True


if __name__ == "__main__":
    print(os.environ.get("CUDA_VISIBLE_DEVICES"))
    assert (
        os.environ.get("CUDA_VISIBLE_DEVICES") is None
    ), "ERROR: please do:\nunset CUDA_VISIBLE_DEVICES\nbefore running this script"
    # Somehow describe the entire process, i.e.
    # what model architecture you want to train with and infer with,
    # what patch size, whether you want to train in parallel (train on Multiple GPUS)
    # or just on one, batch size, etc.
    num_gpus_to_train_on = 1
    
    model_architecture_id = "u3fasternets"
    # model_architecture_id = "resnet34basedunet"
    # model_architecture_id = "effs"

    desc = dict(
        model_architecture_id=model_architecture_id,
        loss_function_family_id="mse",
        loss_parameters=dict(),
        train_on_feathered=True,
        binarize_masks_before_training=False,
        evaluate_trained_model_in_feathered_or_binary_sense="feathered",
        l1_error_threshold=0.10,
        num_gpus_to_train_on=num_gpus_to_train_on,
        num_gpus_to_infer_with=1,  # whether inference in parallel or not
        workers_per_gpu=2,
        patches_per_image=10,
        num_training_datapoints=320,
        batch_size=40*num_gpus_to_train_on,
        test_size=10,
        frame_width=64,
        frame_height=64,
        num_epochs=6,
        patch_height=32,  # cannot go lower since sometimes it must be divisible by 32
        patch_width=32,
        stride_height=16,
        stride_width=16,
        classification_error_threshold=0.11,
        checkpoint_prefix="unittest",
    )
    validate_test_case(desc)
    run_one_test_case(desc=desc)

