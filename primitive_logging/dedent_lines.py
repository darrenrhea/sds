import numpy as np
import sys
from colorama import Fore, Style
import textwrap


def how_many_spaces_could_be_considered_as_indentation(s: str) -> float:
    """
    Blank like like " \n" or "   \n" may be considered as infinite indentation.
    """
    count = 0
    no_nonwhitespace_was_found = True
    for char in s:
        if char not in [" ", "\n"]:
            no_nonwhitespace_was_found = False
        if char in " ":
            count += 1
        else:
            break
    if no_nonwhitespace_was_found:
        return float(np.inf)
    
    return float(count)


def dedent_lines(s: str):
    lines = s.split("\n")
    min_indentation = float(np.inf)
    for index, line in enumerate(lines):
        indent = how_many_spaces_could_be_considered_as_indentation(line)
        print(f"line {index} has {indent=}")

        if indent < min_indentation:
            min_indentation = indent
    
    if np.isinf(min_indentation):
        indentation_to_remove = 0
    else:
        indentation_to_remove = int(min_indentation)
    print(f"{indentation_to_remove=}")

    new_lines = []
    for line in lines:
        a = min(len(line), indentation_to_remove)
        new_line = line[a:]
        new_lines.append(
            new_line
        )
    return "\n".join(new_lines)

