import argparse
from pathlib import Path
import pprint as pp
import textwrap
from infer_arbitrary_frames import (
     infer_arbitrary_frames
)


def ifigd_infer_frames_in_given_directory_cli_tool():
    """
    Given a final model ID, input directory, and output directory, this function will
    infer all the frames
    ending in _original.png or _original.jpg
    from the input directory and save the result as
    grayscale to the output directory.
    """
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """\
            Infer arbitrary frames from input directory and save to output directory."
            """
        ),
        usage=textwrap.dedent(
            f"""\
            python {__file__} -i /path/to/input_dir -o /path/to/output_dir -m final_model_id
            """
        ),
    )
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='Path to the input directory containing original frames.')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Path to the output directory to save inferred frames.')
    parser.add_argument('-m', '--final_model_id', type=str, required=True, help='Final model ID to use for inference.')

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)
    assert input_dir.is_dir(), "Input directory does not even exist."
    final_model_id = args.final_model_id

    output_dir.mkdir(exist_ok=True, parents=True)
    input_paths = [
        x
        for x in input_dir.glob("*_original.png")
    ] + [
        x
        for x in input_dir.glob("*_original.jpg")
    ]

    list_of_input_and_output_file_paths = [
        (
            x,
            output_dir / f"{x.stem[:-len('_original')]}_{final_model_id}.png"
        )
        for x in input_paths
    ]

    pp.pprint(list_of_input_and_output_file_paths)

    infer_arbitrary_frames(
        final_model_id=final_model_id,
        list_of_input_and_output_file_paths=list_of_input_and_output_file_paths
    )

    print("If you want to see them, do this:")

    for input_path, output_path in list_of_input_and_output_file_paths:
        print(f"pri {input_path}")
        print(f"pric {input_path} {output_path}")

