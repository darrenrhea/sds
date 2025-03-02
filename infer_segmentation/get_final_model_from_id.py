from color_print_json import (
     color_print_json
)
from is_sha256 import (
     is_sha256
)
from print_yellow import (
     print_yellow
)
import subprocess
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
from pathlib import Path
from FinalModel import FinalModel
import better_json as bj

from model_architecture_info import get_valid_model_architecture_family_ids

from functools import lru_cache 


@lru_cache(maxsize=100)
def get_final_model_from_id(final_model_id):
    assert isinstance(final_model_id, str), f"final_model_id must be a string, not {final_model_id}"
    assert len(final_model_id) >= 3, "final_model_id must be at least 3 characters long, like Za7, that is 52 * 62 * 62 = 199888 possible model ids"
    for i in range(1, len(final_model_id)):
        assert final_model_id[i] in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    
    if is_sha256(final_model_id):
        print_yellow("This is a sha256 style final_model_id")
        json_path = get_file_path_of_sha256(
            sha256=final_model_id
        )
        
    else:
        subprocess.run(
            [
                "git",
                "pull",
            ],
            cwd=Path("~/r/final_models").expanduser(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        json_path = Path(
            f"~/r/final_models/models/{final_model_id}.json5"
        ).expanduser()

    jsonable = bj.load(json_path)
    
    color_print_json(jsonable)

    for k in [
        "model_architecture_family_id",
        "pytorch_weights_file",
        "original_width",
        "original_height",
        "patch_width",
        "patch_height",
        "patch_stride_width",
        "patch_stride_height",
        "pad_height",
        "otherinfo",
    ]:
        assert k in jsonable, f"json {json_path} does not have key {k}"

    model_architecture_family_id = jsonable["model_architecture_family_id"]
    assert model_architecture_family_id in get_valid_model_architecture_family_ids()

    weights = jsonable["pytorch_weights_file"]
    
    if "sha256" in weights:
        weights_file_sha256 = jsonable["pytorch_weights_file"]["sha256"]

        weights_file_path = get_file_path_of_sha256(
            sha256=weights_file_sha256
        )
    elif "local_path" in weights:
        weights_file_path = Path(weights["local_path"]).resolve()
        assert weights_file_path.exists(), f"weights_file_path {weights_file_path} does not exist"
    else:
        raise Exception(f"weights {weights} does not have local_abs_path or sha256")

    original_width = int(jsonable["original_width"])
    original_height = int(jsonable["original_height"])
    patch_width = int(jsonable["patch_width"])
    patch_height = int(jsonable["patch_height"])
    patch_stride_width = int(jsonable["patch_stride_width"])
    patch_stride_height = int(jsonable["patch_stride_height"])
    pad_height = int(jsonable["pad_height"])

    final_model = FinalModel(
        model_architecture_family_id=model_architecture_family_id,
        weights_file_path=weights_file_path,
        original_width=original_width,
        original_height=original_height,
        patch_width=patch_width,
        patch_height=patch_height,
        patch_stride_width=patch_stride_width,
        patch_stride_height=patch_stride_height,
        pad_height=pad_height
    )
    return final_model

