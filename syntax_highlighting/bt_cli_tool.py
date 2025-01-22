from print_syntax_highlighted_python import print_syntax_highlighted_python
from pathlib import Path
import argparse

def bt_cli_tool():
    """
    competitor to bat
    """
    argparser = argparse.ArgumentParser(description='bt_cli_tool')
    argparser.add_argument('file', help='file to print syntax highlighted')
    args = argparser.parse_args()
    file_path = Path(args.file).resolve()
    assert file_path.is_file(), f"{file_path} is not a file"

    source_code = file_path.read_text()
    extension  = file_path.suffix
    
    language = None

    if extension == ".py":
        language = "python"
    elif extension == ".json":
        language = "json"
    if language == "python":
        print_syntax_highlighted_python(source_code)
    else:
        print(source_code)
