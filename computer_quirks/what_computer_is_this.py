from uname_dash_n import (
     uname_dash_n
)
def what_computer_is_this():
    valid_computer_names = [
        "aang",
        "appa",
        "arya",
        "grogu",
        "jerry",
        "korra",
        "lam",
        "loki",
        "morty",
        "rick",
        "squanchy",
        "dockercontainer",
    ]

    uname_dash_n_to_computer_name = {
        "aang": "aang",
        "appa": "appa",
        "arya": "arya",
        "grogu": "grogu",
        "lam": "lam",
        "loki": "loki",
        "rick": "rick",
        "jerry": "jerry",
        "korra": "korra",
        "morty": "morty",
        "squanchy": "squanchy",  # sudo scutil --set HostName squanchy
    }

    # Run subprocess to execute the 'uname -n' command
    uname_dash_n_result = uname_dash_n()
    
    computer_name = uname_dash_n_to_computer_name.get(
        uname_dash_n_result,
        "dockercontainer"
    )

    assert (
        computer_name in valid_computer_names
    ), f"Computer name {computer_name} is not in the list of valid computer names: {valid_computer_names}"

    return computer_name

