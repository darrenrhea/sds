from load_json_file import (
     load_json_file
)
# import jstyleson  # WE used to use this, but it does not handle JSON5 not maintained
# maybe use pyjson5 instead, it is apparently fast and actually maintained
import json
from io import TextIOWrapper
from typing import Union
import jsonlines
from pathlib import Path
import pyjson5
from dump_as_jsonlines import dump_as_jsonlines
from color_print_json import color_print_json
from save_jsonable import save_jsonable


# define what from better_json import * will import:
__all__ = [
    "color_print_json",
    "read_last_byte_of_file",
    "append_one_object_to_jsonlines_file",
    "load_jsonlines",
    "load_json_file",
    "dump_as_jsonlines",
    "save_jsonable",
]



def read_last_byte_of_file(file_handle):
    """
    Reads the last unicode character of a file,
    or returns None if the file is empty.
    """

    file_handle.seek(0, 2)  # go to 0 from the end of the file
    file_length = file_handle.tell()
    
    if file_length == 0:
        file_handle.seek(0)  # go to 0 from the start of the file
        return None

    file_handle.seek(1, 2)  # Move one character back
    return file_handle.read(1)  # Read the last character



def append_one_object_to_jsonlines_file(
    obj: dict,
    jsonlines_path: Path
):
    """
    If the file does not exist, it is created.
    adds the given jsonable python object obj
    to the end of a jsonlines file.
    """
    assert isinstance(obj, dict), "obj must be a dict"
    
    # watch, that 'r+b' thing is important, r might be random
    with open(str(jsonlines_path), 'a+b',  buffering=0) as writer:
        print(f"{writer.tell()=}")
        print(type(writer))
        last_char = read_last_byte_of_file(file_handle=writer)
        print(f"The last character of the file is: {last_char}")
        writer.seek(3, 2)  # go to 0 from the end of the file
        print(f"{writer.tell()}")
        if last_char is None:  # file is empty
            pass
        elif last_char != b"\n":
            print("last byte was not newline, adding newline")
            writer.write("\n".encode('utf-8'))
        else: # apparently the last_char is already "\n", nothing to do
            pass
        writer.write(json.dumps(obj).encode('utf-8'))
        writer.write("\n".encode('utf-8'))



def load_jsonlines(jsonlines_path: Path):
    """
    Loads an entire jsonlines file into memory at once.
    """
    assert (
        str(jsonlines_path)[-6:] == ".jsonl"
    ), f"ERROR: jsonlines file must end in .jsonl, but\n   {jsonlines_path}\n does not end in .jsonl"

    jsonable = []
    with jsonlines.open(jsonlines_path, 'r') as reader:
       for obj in reader:
           jsonable.append(obj)
    return jsonable


# for bj.load, this abbreviation:
def load(
    path_or_string_or_fp: Union[Path, str, TextIOWrapper]
):
    """
    Loads a JSON / JSON5 JSON-with-comments file into RAM/memory all at once.
    Works with JSON5, JSON with comments, and, of course, plain old JSON.
    """
    return load_json_file(path_or_string_or_fp)


def load_json_string(bytes_or_string):
    """
    Given a string or bytes or bytearray that can be interpreted as JSON,
    """
    assert (
        isinstance(bytes_or_string, str)
        or
        isinstance(bytes_or_string, bytes)
    ), "better_json.loads takes in only a bytes or a string that can be interpreted as JSON, possibly with C-style comments"
    
    if isinstance(bytes_or_string, str):
        # jsonable = jstyleson.loads(bytes_or_string)
        jsonable = pyjson5.decode(bytes_or_string, None, False)

    elif isinstance(bytes_or_string, bytes) or isinstance(bytes_or_string, bytearray):
        jsonable = pyjson5.decode(bytes_or_string.decode("utf-8"))
    else:
        raise Exception("must be str or bytes or bytearrray")
    return jsonable


# for bj.loads, this abbreviation:
def loads(bytes_or_string):
    return load_json_string(bytes_or_string)


# for bj.dump use, this abbreviation:
def dump(fp, obj, indent=4, sort_keys=False):
    return save_jsonable(fp=fp, obj=obj, indent=indent, sort_keys=sort_keys)


def dump_to_string(
    obj,
    indent:int = 4,
    separators=(", ", ": ")
) -> str: 
    return json.dumps(
        obj=obj,
        indent=4,
        separators=(", ", ": ")
    )

# abbreviation for bj.dumps:
def dumps(
    obj,
    indent=4,
    separators=(", ", ": ")
) -> str:
    return dump_to_string(
        obj=obj,
        indent=indent,
        separators=separators
    )


def save(fp, obj, indent=4):
    assert (
        isinstance(fp, str)
        or
        isinstance(fp, Path)
        or
        isinstance(fp, TextIOWrapper)
    ), "better_json.open takes in only a Path or a string that can be interpreted as a Path or a TextIO file-object"
    
    if isinstance(fp, TextIOWrapper):
        json.dump(fp=fp, obj=obj, indent=indent)
        return
    else:
        p = Path(fp).expanduser()
        with open(p, "w") as file_pointer:
            json.dump(fp=file_pointer, obj=obj, indent=indent)
            return

