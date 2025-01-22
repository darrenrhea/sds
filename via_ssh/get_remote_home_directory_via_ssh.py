import subprocess
from pathlib import Path

def get_remote_home_directory_via_ssh(sshremote):
    """
    Refactor to standard arguments.
    Current users:
    vscode_remote_cli_tool
    """
    cmd = f"ssh {sshremote} 'printf \"%s\" $HOME'"
    remotehome = subprocess.check_output(cmd, shell=True, encoding="utf-8")
    assert Path(remotehome).is_absolute()
    return remotehome