
"""
People need to be able to specify which loss function to use
at the command line via the --loss option.
Grossly, the choices are:
bce, i.e. BinaryCrossEntropy,
mse, i.e. MeanSquaredError, and
awl, i.e. Adaptive Wing Loss

However, we are going to have to soon move to something more complicated
than just a string identifier, because these three are really more like
the names of three families of loss functions, and we need to be able
speficify additional floating point parameters to get down to a specific member of the family.
For instance, it has been useful to have "class weights" for the bce loss.
Adaptive Wing Loss was originally designed for facial keypoint detection,
where the keypoint locations are tiny gaussian bumps in a sea of background,
so the hyperparameters for Adaptive Wing Loss are tuned for that.
"""

valid_loss_function_family_ids = [
    "unweighted_l1",  # Unweighted L1 Loss
    "class_weighted_bce",  # Class Weighted Binary Cross Entropy
    "bce_dice_loss",  # Binary Cross Entropy + Dice Loss
    "mse",  # Mean Squared Error
    "awl",  # Adaptive Wing Loss
]

map_from_loss_function_family_id_to_loss_name = dict(
    class_weighted_bce="Class Weighted Binary Cross Entropy",
    bce_dice_loss="Binary Cross Entropy + Dice Loss",
    mse="Mean Squared Error",
    awl="Adaptive Wing Loss",
)


def validate_loss_parameters(loss_function_family_id, loss_parameters):
    """
    Some loss function families require additional parameters to get
    to a specific member of the family.
    This validates that the parameters are present and valid.
    """
    if loss_function_family_id == "unweighted_l1":
        pass
    elif loss_function_family_id == "class_weighted_bce":
        assert "weight_on_negative" in loss_parameters, "weight_on_negative must be specified if you are going to use class_weighted_bce loss"
        weight_on_negative = loss_parameters["weight_on_negative"]
        assert isinstance(weight_on_negative, float)
        assert 0.0 < weight_on_negative and weight_on_negative < 1.0
    elif loss_function_family_id == "bce_dice_loss":
        assert "bce_weight" in loss_parameters, "bce_weight must be specified if you are going to use bce_dice_loss"
        bce_weight = loss_parameters["bce_weight"]
        assert isinstance(bce_weight, float)
        assert 0.0 < bce_weight and bce_weight < 100.0
    elif loss_function_family_id == "awl":
        # TODO: allow theta alpha and omega to be specified
        pass
    elif loss_function_family_id == "duat":
        pass
    elif loss_function_family_id == "mse":
        pass  # no parameters needed for Mean Square Error mse_loss
    else:
        raise Exception(f"unknown loss function family id: {loss_function_family_id}")
    