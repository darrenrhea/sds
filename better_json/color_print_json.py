from ptse import (
     ptse
)
import json
from pygments import highlight, lexers, formatters


def color_print_json(jsonable):
    """
    formats with 4 space indent,
    then syntax highlights then prints a jsonable python object.
    """
    formatted_json = json.dumps(
        obj=jsonable,
        indent=4,
        separators=(", ", ": ")
    )
    colorful_json = highlight(
        code=formatted_json,
        lexer=lexers.JsonLexer(),
        formatter=formatters.TerminalFormatter()
    )
    # we print to stderr:
    ptse(colorful_json)

