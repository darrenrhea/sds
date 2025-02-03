import json
from io import TextIOWrapper
from pathlib import Path
from collections import OrderedDict


def save_jsonable(fp, obj, indent=4, sort_keys=False):
    """
    You cannot dump comments because they are gone,
    so the standard library json is fine.
    TODO: make a variant that dumps to the best JSON5,
    i.e. keys without quotes, commas after all, etc.
    """
    assert (
        isinstance(obj, dict)
        or
        isinstance(obj, OrderedDict)
        or
        isinstance(obj, list)
    ), "obj must be a dict or a list"
    assert (
        isinstance(fp, str)
        or
        isinstance(fp, Path)
        or
        isinstance(fp, TextIOWrapper)
    ), "save_jsonable takes fp which must be either a Path or a string that can be interpreted as a Path or a TextIO file-object"
    
    if isinstance(fp, TextIOWrapper):
        json.dump(fp=fp, obj=obj, indent=indent, sort_keys=sort_keys)
        return
    else:
        p = Path(fp).expanduser()
        with open(p, "w") as file_pointer:
            json.dump(fp=file_pointer, obj=obj, indent=indent, sort_keys=sort_keys)
            return
