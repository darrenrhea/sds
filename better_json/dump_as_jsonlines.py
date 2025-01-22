import json
from io import TextIOWrapper
from pathlib import Path


def dump_as_jsonlines(fp, obj):
    """
    JSONLines, also known as newline-delimited JSON,
    is a convenient format for storing structured data that
    at the outermost level is a list of objects.

    In particular, it is strictly better than csv or tsv files
    which have no standard escaping rules and no way to store nested data.
    TODO: JSON5Lines?
    """
    assert (
        isinstance(obj, list)
    ), "ERROR: Cannot dump as jsonlines unless the outermost level is a list"

    assert (
        isinstance(fp, str)
        or
        isinstance(fp, Path)
        or
        isinstance(fp, TextIOWrapper)
    ), "better_json.open takes in only a Path or a string that can be interpreted as a Path or a TextIO file-object"
    
    if isinstance(fp, TextIOWrapper):
        out_fp = fp
    else:
        p = Path(fp).expanduser()
        out_fp = open(p, "w")
    
    for entry in obj:
        line = json.dumps(entry)
        out_fp.write(line + "\n")
    
    out_fp.close()

