from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)

from prii import prii

from infer_arbitrary_frames import (
     infer_arbitrary_frames
)
from pathlib import Path


def test_infer_arbitrary_frames_1():
    folder = Path("/shared/temp")
    folder.mkdir(exist_ok=True)
    clip_id = "nfl-59773-skycam-ddv3"
    clip = [
        dict(
            frame_index=0,
            sha256="a4165e46301a660ab0b4018e5c66804b7b69f5c6f58c679e8bf650425bb5b3da",
        ),

        dict(
            frame_index=100,
            sha256="67dff56d7c07e16460f073fc9c1a386408b798d82a9f0e91157c8bf8fa496375",
        ),
        dict(
            frame_index=200,
            sha256="cf784af460dc034f675756cd56b0f5826859255fb2c5b08038893e5d28ee1e34",
        ),
    ]
    
    list_of_input_and_output_file_paths = []
    for dct in clip:
        frame_index = dct["frame_index"]
        sha256 = dct["sha256"]

        input_file_path = get_file_path_of_sha256(
            sha256=sha256
        )
        output_file_path = folder / f"{clip_id}_{frame_index:06d}_nonfloor.png"

        pair = (
            input_file_path,
            output_file_path,
        )
        list_of_input_and_output_file_paths.append(pair)
    
    
    final_model_id = "148dd52080006cf62e0cbb60c8011f24326e0f0c8d10c63e05c5fd5105f8fddd"

    infer_arbitrary_frames(
        final_model_id=final_model_id,
        list_of_input_and_output_file_paths=list_of_input_and_output_file_paths
    )

    for input_file_path, output_file_path in list_of_input_and_output_file_paths:
        prii(input_file_path)
        prii(output_file_path)


if __name__ == "__main__":
    test_infer_arbitrary_frames_1()