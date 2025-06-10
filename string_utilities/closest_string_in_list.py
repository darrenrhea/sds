import textwrap
from print_yellow import (
     print_yellow
)
from print_red import (
     print_red
)
from typing import List
from Levenshtein import distance as levenshtein_distance



def closest_string_in_list(
    s: str,
    valid_strings: List[str],
    crash_on_inexact: bool = True
) -> str:
    """
    Return the closest string in *valid_strings* to *s* (by Levenshtein).

    If *s* is already present in *valid_strings*, it is returned immediately.
    Otherwise the closest match is printed and returned.
    """
    assert len(valid_strings) > 0, "valid_strings must not be empty"
    
    for candidate in valid_strings:
        assert isinstance(candidate, str), "valid_strings must contain only strings"
    
    if s in valid_strings:
        return s

    # Find the string with the minimal distance
    best_match = None
    best_distance = float("inf")
    for candidate in valid_strings:
        d = levenshtein_distance(s, candidate)
        if d < best_distance:
            best_distance, best_match = d, candidate
            if best_distance == 0:            # exact match (unlikely here)
                break
    
    print_yellow("valid_strings are:")
    sorted_valid_strings = sorted(valid_strings)
    for candidate in sorted_valid_strings:
        print_yellow(f'"{candidate}"')
    print_red(
        textwrap.dedent(
            f"""\
            ERROR: Did you maybe mean to say '{best_match}'?
            you gave
                {s}
            which is not valid,
            but is only levenshtein distance={best_distance} away
            from the valid possibility
                '{best_match}'.
            """
        )
    )
    if crash_on_inexact:
        raise Exception()
    
    return best_match