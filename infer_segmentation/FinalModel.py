from pathlib import Path
from model_architecture_info import get_valid_model_architecture_family_ids

class FinalModel(object):
    """
    A FinalModel defines EVERY DECISION that is needed to make masks,
    including weirder things like patch_stride_width
    and patch_stride_height, which might cause quality to go up
    by much averaging of patches.
    """
    def __init__(self,
        model_architecture_family_id: str,
        weights_file_path: Path,
        original_width: int,
        original_height: int,
        patch_width: int,
        patch_height: int,
        patch_stride_width: int,
        patch_stride_height: int,
        pad_height: int
    ):
        self.model_architecture_family_id = model_architecture_family_id
        assert model_architecture_family_id in get_valid_model_architecture_family_ids()

        self.weights_file_path = weights_file_path
        assert self.weights_file_path.exists(), f"weights_file_path {weights_file_path} does not exist."
        self.original_width = original_width
        self.original_height = original_height
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.patch_stride_width = patch_stride_width
        self.patch_stride_height = patch_stride_height
        self.pad_height = pad_height


