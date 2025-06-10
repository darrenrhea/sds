from print_green import (
     print_green
)
from closest_string_in_list import (
     closest_string_in_list
)
import pytest


def test_closest_string_in_list_1():
    valid_strings = ["apple", "banana", "orange", "grape"]
    s = "orgne"
    with pytest.raises(Exception):
        closest_string_in_list(
            s=s, valid_strings=valid_strings, crash_on_inexact=True
        )

def test_closest_string_in_list_2():
    valid_strings = ["apple", "banana", "orange", "grape"]
    s = "orgne"
    ans = closest_string_in_list(
        s=s,
        valid_strings=valid_strings,
        crash_on_inexact=False,
    )
    assert ans == "orange", f"Expected 'orange', but got '{ans}'"

if __name__ == "__main__":
    test_closest_string_in_list_2()
    test_closest_string_in_list_2()
    print_green("Test completed successfully.")
