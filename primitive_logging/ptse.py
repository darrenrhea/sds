import sys


def ptse(s: str):
    """
    Programs have comments/logging/debug-printouts, but then
    they also have output.  The output is the only thing that should be
    printed to stdout.
    """
    
    print(
        s,
        file=sys.stderr,
    )
