import better_json as bj
from pathlib import Path


def test_loads_str():
    the_bytes = '{"cat": 3}'
    jsonable = bj.loads(the_bytes)
    should_be = dict(cat=3)
    assert jsonable == should_be

def test_loads_bytes():
    the_bytes = b'{"cat": 3}'

    jsonable = bj.loads(the_bytes)
    should_be = dict(cat=3)
    assert jsonable == should_be

def test_load_0():
    with open("fixture.json", "r") as fp:
        jsonable = bj.load(fp)
    assert jsonable["cat"] == 3


def test_load_1():
    jsonable = bj.load("fixture.json")
    assert jsonable["cat"] == 3


def test_load_2():
    jsonable = bj.load(Path("fixture.json"))
    assert jsonable["cat"] == 3


def test_save_0():
    jsonable = dict(cat=3)
    bj.save(fp=Path("temp.json"), obj=jsonable)
    with open("temp.json", "r") as fp:
        s = fp.read()
    
    assert s == """\
{
    "cat": 3
}"""



def test_save_expanduser():
    jsonable = dict(cat=3)
    
    bj.save(fp="~/temp.json", obj=jsonable)

    with open("temp.json", "r") as fp:
        s = fp.read()
    
    assert s == """\
{
    "cat": 3
}"""

