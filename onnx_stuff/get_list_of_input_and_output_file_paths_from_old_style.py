from typing import List
from pathlib import Path
import glob
import numpy as np


def get_list_of_input_and_output_file_paths_from_old_style(
    fn_input: str,
    out_dir: Path,
    model_id_suffix: str,
    limit: int = 0
) -> List[str]:
    """
    All inferers will take in a list of input and output file paths,
    which is what image to infer on and where to write the output.
    """
    assert isinstance(model_id_suffix, str), "model_id_suffix should be a string"
    assert isinstance(fn_input, str), "fn_input should be a string"
    assert isinstance(out_dir, Path), "out_dir should be a Path"
    assert out_dir.is_dir(), f"{out_dir} is not an extant directory"


    fn_frames = []
    if '**' in fn_input:
        tokens = fn_input.split('/')
        path = '/'.join(tokens[:-1])
        fn_frames = [str(x) for x in Path(path).rglob(tokens[-1].replace('**', '*'))]
    elif '*' in fn_input:
        fn_frames = glob.glob(fn_input)
    else:
        fn_frames = [fn_input]

    fn_frames = [
        fn.replace(
            '_nonwood.png', '.jpg'
        ).replace(
            '_nonfloor.png', '.jpg'
        )
        for fn in fn_frames
    ]

    if limit > 0:
        np.random.seed(1337)
        perm = np.random.permutation(len(fn_frames))[:limit]
        fn_frames = [fn_frames[j] for j in perm]

    list_of_input_and_output_file_paths = []
    for input_file_path_str in fn_frames:
        input_file_path = Path(input_file_path_str)
        if input_file_path.name.endswith("_original.png"):
            annotation_id = input_file_path.name[:-13]
            print(f"{annotation_id=}")
        else:
            annotation_id = input_file_path.stem
        
        output_file_path = out_dir / f"{annotation_id}{model_id_suffix}.png"
        list_of_input_and_output_file_paths.append(
            (input_file_path, output_file_path)
        )
        assert input_file_path.is_file(), f"{input_file_path} is not a file"

    return list_of_input_and_output_file_paths

