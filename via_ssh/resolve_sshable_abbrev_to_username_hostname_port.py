import paramiko 
from pathlib import Path
import pprint as pp
from typing import Tuple


def resolve_sshable_abbrev_to_username_hostname_port(
    sshable_abbrev: str = None,
    remote_user: str = None,
    remote_host: str = None,
    remote_ssh_port: int = 22,
    verbose: bool = False
) -> Tuple[str, str, int]:
    
    """
    If you do not given an sshable_abbrev, it just passes things through.

    If you do give an sshable_abbrev, it looks up the hostname, port, and username.

    This expects that you have an entry on the executing machine
    in the executing user's home folder in the
    ~/.ssh/config file
    that points to that information as well as other information
    like agent forwarding and the location of
    the public / private key pair.
    """
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
        if "hostname" not in conf or "user" not in conf:
            print(f"Failed to lookup sshable abbreviation {sshable_abbrev=}")
            return (None, None, None)
        
        hostname = conf["hostname"]
        port = int(conf.get("port", 22))
        username = conf["user"]
    else:
        username = remote_user
        hostname = remote_host
        port = remote_ssh_port
    
    assert isinstance(hostname, str), f"ERROR: {hostname=} is not a string"
    assert isinstance(port, int) and port < 65536, f"ERROR: {port=} is not an int or is too large"
    assert isinstance(username, str), f"ERROR: {username=} is not a string"
    
    return (
        username,
        hostname,
        port,
    )