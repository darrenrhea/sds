# import jstyleson  # WE used to use this, but it does not handle JSON5 not maintained
# maybe use pyjson5 instead, it is apparently fast and actually maintained
import json
from pygments import highlight, lexers, formatters


def color_print_json(jsonable):
    """
    syntax highlights then prints a jsonable python object
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
    print(colorful_json)

