import sys
from loss_info import valid_loss_function_family_ids, validate_loss_parameters

def validate_test_case(desc):
    assert (
        isinstance(desc, dict)
    ), "ERROR: desc must be a dictionary"
    required_keys = [
        "train_on_feathered",
        "loss_function_family_id",
        "loss_parameters",
        "checkpoint_prefix",
        "num_gpus_to_train_on",
        "batch_size",
        "evaluate_trained_model_in_feathered_or_binary_sense",
        "num_epochs",
        "frame_width",
        "frame_height",
        "patch_width",
        "patch_height",
        "stride_width",
        "stride_height",
        "num_training_datapoints",
        "binarize_masks_before_training",
    ]
    for key in required_keys:
        assert (
            key in desc
        ), f"ERROR: desc must have the key {key}"
    
    train_on_feathered = desc["train_on_feathered"]
    
    assert (
        isinstance(train_on_feathered, bool)
    ), "ERROR: train_on_feathered must be a boolean"

    loss_function_family_id = desc["loss_function_family_id"]

    assert isinstance(loss_function_family_id, str), "ERROR: loss_function_family_id must be a string"
    
    assert (
        loss_function_family_id in valid_loss_function_family_ids
    ), f"ERROR: loss_function_family_id must be one of {valid_loss_function_family_ids}"

    loss_parameters = desc["loss_parameters"]
    validate_loss_parameters(loss_function_family_id, loss_parameters)

    checkpoint_prefix = desc["checkpoint_prefix"]
    assert isinstance(checkpoint_prefix, str), "ERROR: checkpoint_prefix must be a string"
    num_gpus_to_train_on = desc["num_gpus_to_train_on"]
    batch_size = desc["batch_size"]

    assert batch_size >= 3, "ERROR: batch_size must be at least 3 because a lot of these models use batchnorm"

    if batch_size < 3 * num_gpus_to_train_on:
        print("WARNING: if you are training on 2 GPUs, the batch size must be at least 4")
        print("setting batch_size 2 or 3 will give the mysterious error:")
        print("ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 960, 1, 1])")
        print("more generally, batch_size must be >= 3 * num_GPUs")
        sys.exit(1)

    evaluate_trained_model_in_feathered_or_binary_sense = desc["evaluate_trained_model_in_feathered_or_binary_sense"]

    if evaluate_trained_model_in_feathered_or_binary_sense == "feathered":
        assert "l1_error_threshold" in desc, "ERROR: if you are evaluating the trained model in the feathered sense, you must specify the l1_error_threshold"
        l1_error_threshold = desc["l1_error_threshold"]
        assert isinstance(l1_error_threshold, float), "ERROR: l1_error_threshold must be a float"
        assert l1_error_threshold > 0.0, "ERROR: l1_error_threshold must be positive"

