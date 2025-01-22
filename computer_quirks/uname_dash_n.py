import subprocess


def uname_dash_n():
    """
    A function that returns the output of
    running the 'uname -n' command
    """
    # Run subprocess to execute the 'uname -n' command:
    computer_name = subprocess.run(
        args=[
            'uname',
            '-n'
        ],
        capture_output=True,
        text=True
    ).stdout.strip()
    return computer_name

