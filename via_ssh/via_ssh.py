from pathlib import Path
import paramiko 
from typing import List, Tuple

def download_files(
    machine: str,
    remote_src_local_dest_path_pairs: List[Tuple[Path, Path]]
):
    """
    This expects that you have an entry for machine in your
    ~/.ssh/config file
    that points it to the public / private key pair.
    """
    my_config = paramiko.SSHConfig()
    
    my_config.parse(
        open(
            Path('~/.ssh/config').expanduser()
        )
    )
    conf = my_config.lookup(machine)
    # pp.pprint(conf)
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(
        hostname=conf["hostname"],
        port=conf['port'],
        username=conf["user"]
    )
    sftp = ssh_client.open_sftp()
    for remotepath, localpath in remote_src_local_dest_path_pairs:
        print(f"Downloading {machine}:{remotepath} to {localpath}")
        assert isinstance(remotepath, Path)
        assert isinstance(localpath, Path)
        assert localpath.is_absolute()
        assert remotepath.is_absolute()
        sftp.get(
            remotepath=str(remotepath),
            localpath=str(localpath)
        )
    sftp.close()
    ssh_client.close()
