from resolve_sshable_abbrev_to_username_hostname_port import (
     resolve_sshable_abbrev_to_username_hostname_port
)
import paramiko 
from pathlib import Path
import pprint as pp
from typing import Optional, Tuple
import stat


def remote_file_exists(
    remote_abs_file_path_str: str,
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
    if sshable_abbrev is not None:
        my_config = paramiko.SSHConfig()
        
        my_config.parse(
            open(
                Path("~/.ssh/config").expanduser()
            )
        )
        conf = my_config.lookup(sshable_abbrev)
        if verbose:
             pp.pprint(conf)
        if "port" not in conf:
            print(f"Failed to lookup sshable abbreviation {sshable_abbrev=}")
            return (None, None)
        
        hostname = conf["hostname"]
        port = conf["port"]
        username = conf["user"]
    else:
        hostname = remote_host
        port = remote_ssh_port
        username = remote_user
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh_client.connect(
            hostname=hostname,
            port=port,
            username=username
        )
        sftp = ssh_client.open_sftp()
        connection_succeeded = True
    except Exception as e:
        connection_succeeded = False
    
    if not connection_succeeded:
        return False, None
    
    try:
        filestat = sftp.stat(remote_abs_file_path_str)
        mode = filestat.st_mode
        is_file = stat.S_ISREG(mode)
    except Exception as _:
        is_file = False
   
    sftp.close()
    ssh_client.close()
    return connection_succeeded, is_file
