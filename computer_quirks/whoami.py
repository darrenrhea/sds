import subprocess

def whoami() -> str:
    """
    A function that returns the output of the whoami command
    """
    # Run subprocess to execute the 'whoami' command:
    whoami = subprocess.run(
        args=[
            'whoami',
        ],
        capture_output=True,
        text=True
    ).stdout.strip()
    assert (
        whoami in {
            "anna",
            "annaayzenshtat",
            "cav",
            "chaz",
            "darren",
            "darrenrhea",
            "drhea",
            "felix",
            "julien",
            "mathieu",
            "root",
        }
    ), f"ERROR: who is this: {whoami=}? Never heard of them."
    return whoami
