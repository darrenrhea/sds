from what_computer_is_this import (
     what_computer_is_this
)
from pathlib import Path
from typing import Optional
from functools import lru_cache 


@lru_cache(maxsize=100, typed=False)
def get_the_large_capacity_shared_directory(
    computer_name: Optional[str] = None
) -> Path:
    """
    If called with no arguments, i.e. None,
    this function returns the large capacity shared directory
    on the current computer.

    Or you can pass in a computer name and it will return the large capacity shared directory on that computer.

    Each computer hopefully has a shared directory that everyone can access,
    backed by a large capacity hard drive or NAS.
    TODO: fix for computers that use someone's home directory.
    Won't work if it is executed on a different machine.
    """
    # Defensive copy to know if was called with None.  computer_name mutates below.
    input_computer_name = computer_name
 
    computers_on_which_we_have_a_slash_shared_dir = [
        "appa",
        "grogu",
        "loki",
        "rick",
        "jerry",
        "morty"
    ]
    if computer_name is None:
        computer_name = what_computer_is_this()
    # although others have a NAS, only lam has a NAS that is effectively permanently mounted enough to be used:
    if computer_name == "lam":
        shared_dir = Path("/media/drhea/muchspace")
    elif computer_name == "squanchy":
        shared_dir = Path("/Users/awecom/a")
    elif computer_name in computers_on_which_we_have_a_slash_shared_dir:
        shared_dir = Path("/shared")
    elif computer_name == "aang":
        shared_dir = Path("/Users/darrenrhea/a").expanduser()
    elif computer_name == "arya":
        shared_dir = Path("/Users/annaayzenshtat/a").expanduser()
    elif computer_name == "korra":
        shared_dir = Path("/Users/anna/a").expanduser()
    else:
        raise Exception(
            f"I don't know what the large capacity shared directory is for the computer {computer_name}"
        )
    
    if input_computer_name is None:  # if the caller was asking about the shared directory on their current computer, you can check it exists easily:
        assert shared_dir.is_dir(), f"ERROR: {shared_dir=} is not an extant directory"
    return shared_dir

    