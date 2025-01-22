from get_clicks_on_image_with_color_confirmation import (
     get_clicks_on_image_with_color_confirmation
)
from collections import OrderedDict

import uuid
from prii import (
     prii
)
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from color_print_json import (
     color_print_json
)

from pathlib import Path
import better_json as bj


def attempt_to_annotate_a_color_correction_data_point(
    color_correction_context_id: str,
    domain_image_sha256: str,
    codomain_image_sha256: str,
    domain_image_path: Path,
    codomain_image_path: Path,
    instructions_str: str,
):
    """
    For a given color_correction_context_id,
    i.e. the id of a situation for which you think training a regression might be appropriate,
    this function will prompt the user to click on the domain image and then the codomain image.
    This matches points in the domain image to points in the codomain image.
    """

    domain_points = get_clicks_on_image_with_color_confirmation(
        image_path=domain_image_path,
        instructions_str=instructions_str,
    )
    
    if domain_points is None:
        return False
    
    description = input("Please describe the spot that you are clicking on:")
    
    codomain_points = get_clicks_on_image_with_color_confirmation(
        image_path=codomain_image_path,
        instructions_str=instructions_str,
    )
    
    out_dir = Path(
        "~/r/color_correction_data/color_correction_data_points"
    ).expanduser()

    out_path = out_dir / f"{uuid.uuid4()}.json5"

    obj = OrderedDict(
        color_correction_context_id=color_correction_context_id,
        description=description,
        domain_points=domain_points,
        codomain_points=codomain_points,
        domain_image_sha256=domain_image_sha256,
        codomain_image_sha256=codomain_image_sha256,
    )
    bj.dump(obj=obj, fp=out_path)
    print("wrote:")
    color_print_json(obj)
    print(f"to {out_path}")
   

def gdfcc_gather_data_for_color_correction_cli_tool():
    
    # color_correction_contexts = get_all_color_correction_context_ids()

    
    # get_input_with_completion_and_validation(
    #     question: "What is the color correction context id? If you dont have on, press enter",
    #     validator: Callable[[str], bool],
    #     completer=None
    # )

    # color_correction_context_id = "d49366a8-2459-4f33-ab47-843cb4cc0911"
    # color_correction_context_id = "5922a16d-0a20-467e-b72c-6a64ca78fabe"
    color_correction_context_id = "4abb9876-9e07-4b0f-979c-6f51152341b1"
    
    color_correction_context = bj.load(
        f"~/r/color_correction_data/color_correction_contexts/{color_correction_context_id}.json5"
    )

    color_print_json(color_correction_context)

    domain_image_sha256 = color_correction_context["domain"]["sha256"]
    domain_image_file_path = get_file_path_of_sha256(sha256=domain_image_sha256)

    domain_rgb_np_u8 = open_as_rgb_hwc_np_u8(
        image_path=domain_image_file_path
    )
    prii(domain_rgb_np_u8, caption="Here is the domain image:")


    codomain_image_sha256 = color_correction_context["codomain"]["sha256"]
    codomain_image_file_path = get_file_path_of_sha256(sha256=codomain_image_sha256)

    codomain_rgb_np_u8 = open_as_rgb_hwc_np_u8(
        image_path=codomain_image_file_path
    )
    
    prii(codomain_rgb_np_u8, caption="Here is the codomain image:")

    attempt_to_annotate_a_color_correction_data_point(
        color_correction_context_id=color_correction_context_id,
        domain_image_path=domain_image_file_path,
        domain_image_sha256=domain_image_sha256,
        codomain_image_sha256=codomain_image_sha256,
        codomain_image_path=codomain_image_file_path,
        instructions_str="click on the domain image.",
    )




   



if __name__ == "__main__":
    gdfcc_gather_data_for_color_correction_cli_tool()