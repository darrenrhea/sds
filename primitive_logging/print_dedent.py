from dedent_lines import (
     dedent_lines
)
import textwrap

def print_dedent(s):
    """
    Fixes the indentation of a string and prints it.
    """
    # dedented = dedent_lines(s)
    # print(dedented)
    print(textwrap.dedent(s))


