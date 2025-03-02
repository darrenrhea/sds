from print_yellow import (
     print_yellow
)
import os
from pathlib import Path
import subprocess

from colorama import Fore, Style
from get_final_model_from_id import (
     get_final_model_from_id
)
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)


def infer_from_id(
    final_model_id: str,
    model_id_suffix: str,
    frame_ranges_file_path: Path,
    output_dir: Path = None,
):
    """
    TODO: This is goofy because it is Python calling Python as a subprocess.
    """
    # By default, the output_dir is the shared directory on the machine:

    if output_dir is None:
        shared_dir = get_the_large_capacity_shared_directory()
        output_dir = shared_dir / "inferences"

    final_model = get_final_model_from_id(
        final_model_id=final_model_id,
    )

    model_architecture_family_id = final_model.model_architecture_family_id
    weights_file_path = final_model.weights_file_path
    original_width = final_model.original_width
    original_height = final_model.original_height
    patch_width = final_model.patch_width
    patch_height = final_model.patch_height
    patch_stride_width = final_model.patch_stride_width
    patch_stride_height = final_model.patch_stride_height
    pad_height = final_model.pad_height

    # TODO: ad assertions
    current_file_path = Path(__file__).resolve()
    dir_path = current_file_path.parent

    args = [
        "python",
        str(dir_path / "parallel_infer3.py"),
        model_architecture_family_id,
        str(weights_file_path),
        "--original-size",
        f"{original_width},{original_height}",
        "--patch-width",
        f"{patch_width}",
        "--patch-height",
        f"{patch_height}",
        "--patch-stride-width",
        f"{patch_stride_width}",
        "--patch-stride-height",
        f"{patch_stride_height}",
        "--pad-height",
        f"{pad_height}",
        "--model-id-suffix",
        model_id_suffix,
        "--out-dir",
        f"{output_dir}",
        str(frame_ranges_file_path),
    ]

    # least likely to be used:
    # CUDA_VISIBLE_DEVICES = "1"
    CUDA_VISIBLE_DEVICES = "0,1,2"
    print_yellow("The inference bash command we are testing is basically:")
    print_yellow(
        f"export CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} &&\n"
        +
        " \\\n".join(args)
    )

    environ = os.environ.copy()
    environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    completed_process = subprocess.run(
        args=args,
        env=environ,
        # cwd=Path("~/r/major_rewrite").expanduser(),
    )
    if completed_process.returncode != 0:
        print(f"{Fore.RED}parallel_infer3.py process FAILED with exit code {completed_process.returncode}!{Style.RESET_ALL}")
        return False
    
    print(f"{Fore.RED}process COMPLETED SUCCESSFULLY!{Style.RESET_ALL}")

