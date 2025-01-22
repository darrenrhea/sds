from unexpanduser import (
     unexpanduser
)

from pathlib import Path


def test_unexpanduser_1():
       
    home = Path("~").expanduser()
    xy_pairs = [ 
        (
            home / "dog.txt",
            "~/dog.txt",
        ),
        (
            home / "a" / "b",
            "~/a/b",
        ),
        (
            Path("a/b"),
            "a/b",
        ),
        (
            str(home / "dog.txt"),
            "~/dog.txt",
        ),
        (
            str(home / "a" / "b"),
            "~/a/b",
        ),
        (
            home,
            "~",
        ),

    ]
    for x, y in xy_pairs:
        assert unexpanduser(x) == y
    