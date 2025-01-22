from dump_as_jsonlines import (
     dump_as_jsonlines
)
from get_a_temp_file_path import (
     get_a_temp_file_path
)


def test_dump_as_jsonlines_1():
    # does it work when fp is an abs_file_path?
    fp = get_a_temp_file_path()
    dump_as_jsonlines(fp=fp, obj=[{"a": 1}, {"b": 2}])

    should_be = """\
{"a": 1}
{"b": 2}
"""

    assert fp.read_text() == should_be
    