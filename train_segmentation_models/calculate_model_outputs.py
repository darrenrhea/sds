from model_architecture_info import (
    valid_model_architecture_ids,
    first_coordinate_model_architecture_ids,
    second_coordinate_model_architecture_ids,
    dict_model_architecture_ids
)


def calculate_model_outputs(
    model,
    model_architecture_id: str,
    inputs,
    train: bool
) -> dict:
    """
    Different neural network architectures need different handling
    to put (a batch of) inputs through the model to get "the" output tensor.
    
    Neural network models sometimes return several output tensors,
    * sometimes as a tuple,
    * sometimes as a dictionary with god knows what keys.

    This picks the one we care about.
    """
    model_on_gpu = next(model.parameters()).is_cuda
    assert model_on_gpu, "model must be on the GPU"

    assert (
        inputs.is_cuda
    ), "inputs must be on the GPU. Oftentimes this happens because you gave a non-existant gpu via something like CUDA_VISIBLE_DEVICES=12345678"

    assert (
        model_architecture_id in valid_model_architecture_ids
    ), f"model architecture id must be one of {valid_model_architecture_ids}"

    if model_architecture_id == "duat":
        # ---- forward ----
        P1, P2 = model(inputs, train=train)
        dict_of_output_tensors = dict(P1=P1, P2=P2)
    else:
        if model_architecture_id in second_coordinate_model_architecture_ids:
            gt_pre, outputs = model(inputs)
        elif model_architecture_id in first_coordinate_model_architecture_ids:
            outputs, _ = model(inputs)
        else:
            outputs = model(inputs)

        if model_architecture_id in dict_model_architecture_ids and type(outputs) is dict:
            outputs = outputs['final_pred']
        
        dict_of_output_tensors = dict(outputs=outputs)

    return dict_of_output_tensors
  
