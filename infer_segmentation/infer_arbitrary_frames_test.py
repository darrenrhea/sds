
from prii import prii

from infer_arbitrary_frames import (
     infer_arbitrary_frames
)
from pathlib import Path


def test_infer_arbitrary_frames_1():
    final_model_id = "doesitwork"
    folder = Path("/shared/fake_sl/annastyle")
    
    list_of_input_and_output_file_paths = [
        (
            folder / "bos-mia-2024-04-21-mxf_548100_fake341996978071216_original.png",
            Path("a.png"),
        ),
    ]
    infer_arbitrary_frames(
        final_model_id=final_model_id,
        list_of_input_and_output_file_paths=list_of_input_and_output_file_paths
    )
    for input_file_path, output_file_path in list_of_input_and_output_file_paths:
        prii(input_file_path)
        prii(output_file_path)


if __name__ == "__main__":
    test_infer_arbitrary_frames_1()