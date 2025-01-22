import paramiko 
from typing import Optional, Tuple


def raw_remote_get_stat(
    remote_path_str: str,
    remote_user: str = None,
    remote_host: str = None,
    remote_ssh_port: int = 22,
    verbose: bool = False
) -> Tuple[bool, Optional[dict]]:
    """
    Given the string-path to a file or directory or symlink on a remote machine,
    This stats it and returns the stat object if it can, 
    None if connection fails or if the file does not exist.
    
    You could then analyze the stat to determine if it is a directory, file, symlink, pipe etc. via https://docs.python.org/3/library/stat.html
    """

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
    except Exception as _:
        connection_succeeded = False
    
    if not connection_succeeded:
        return False, None
    
    try:
        statinfo = sftp.stat(remote_path_str)
        maybe_stat = dict(
            st_size=statinfo.st_size,
            st_uid=statinfo.st_uid,
            st_gid=statinfo.st_gid,
            st_mode=statinfo.st_mode,
            st_atime=statinfo.st_atime,
            st_mtime=statinfo.st_mtime,
            attr=statinfo.attr
        )
    except Exception as _:
        # path does not exist or user does not have permission to stat it:
        maybe_stat = None
    
    return (connection_succeeded, maybe_stat)
