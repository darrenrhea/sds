def form_checkpoint_prefix(
    other: str,
    model_architecture_family_id: str,
    segmentation_convention: str,
    num_training_points: int,
    patch_width: int,
    patch_height: int,
):
    """
    This is bad.  Lets go back to saying a checkpoint is a JSON
    so all this can fit sanely.

    We have these awfully long string names for the
    checkpoints, the trained model weights that are learned.
    Something should form them regularly so that they are "invertible".

    A machine, should be able to parse them back
    to know:
     
    1. what the model_architecture_family_id was,
    2  the patch width and height
    3. what the segmentation_convention was,
    4. what the num_training_points was,
    5. how much data was trained on,

    """
   

    howmuchdata = f"{num_training_points}frames"
    resolution = f"{patch_width}x{patch_height}"
    checkpoint_prefix = f"{model_architecture_family_id}-{segmentation_convention}-{howmuchdata}-{resolution}-{other}"
    return checkpoint_prefix

