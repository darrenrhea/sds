from resolve_sshable_abbrev_to_username_hostname_port import (
     resolve_sshable_abbrev_to_username_hostname_port
)
import paramiko 
from pathlib import Path
import pprint as pp
from typing import Optional, Tuple
import stat


def remote_directory_exists(
    remote_directory_str: str,
    sshable_abbrev: str = None,
    remote_user: str = None,
    remote_host: str = None,
    remote_ssh_port: int = 22,
    verbose: bool = False
) -> Tuple[Optional[bool], Optional[bool]]:  # see below for why Optional[bool]
    """
    TODO: refactor into remote stat, then analyze the stat to determine if it is a directory, file, symlink, pipe etc. via https://docs.python.org/3/library/stat.html
    
    Basically actual True or False are definitive answers,
    whereas None mean I don't know.

    Returns if connection succeed and if the directory exists.
    
    connection_succeeded is None
    means that it couldn't even
    try to connect because it doesn't know what host to connect to.

    directory_exists is None means that it does not know if the remote directory exists or not,
    probably because network connectivity failed.

    Often we need to determine if a remote directory exists
    on an sshable machine.

    TODO: if it cannot ssh to the machine, how to handle that?
    This expects that you have an entry for machine in your
    ~/.ssh/config file
    that points it to the public / private key pair.
    """
    # all of our remote ssh procedures that aren't raw begin with this:
    username, hostname, port = resolve_sshable_abbrev_to_username_hostname_port(
        sshable_abbrev=sshable_abbrev,
        remote_host=remote_host,
        remote_user=remote_user,
        remote_ssh_port=remote_ssh_port,
    )
    if username is None:
        return None, None
    
        
    if sshable_abbrev is not None:
        username, hostname, port = resolve_sshable_abbrev_to_username_hostname_port(
            sshable_abbrev=sshable_abbrev
        )
    else:
        username = remote_user
        hostname = remote_host
        port = remote_ssh_port
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        if verbose:
            print(f"Checking for the existence of {remote_directory_str} {username=}, {hostname=}, {port=}")
        ssh_client.connect(
            hostname=hostname,
            port=port,
            username=username
        )
        sftp = ssh_client.open_sftp()
        connection_succeeded = True
    except Exception as e:
        print(f"We failed to connect to {hostname=}, {port=}, {username=}")
        connection_succeeded = False
    
    if not connection_succeeded:
        return False, None
    
    try:
        filestat = sftp.stat(remote_directory_str)
        mode = filestat.st_mode
        is_dir = stat.S_ISDIR(mode)
    except Exception as e:
        # it is neither a file not a directory nor symlink nor anything
        is_dir = False
   
    sftp.close()
    ssh_client.close()
    return connection_succeeded, is_dir
