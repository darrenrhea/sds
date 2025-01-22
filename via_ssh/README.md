# via_ssh


## Installation

For now, you install via_ssh by doing this:

```bash
conda activate the_python_you_want_to_install_via_ssh_into

git clone git@github.com:darrerhea/via_ssh

cd via_ssh

pip install -e .
```

We have a number of host machines that we can
ssh into without typing a password because we have
and entry for them in `~/.ssh/config` like:

```
Host lam
    AddKeysToAgent yes
    hostname 192.168.0.5
    port 42841
    user drhea
    identityfile ~/.ssh/id_rsa
    ForwardAgent yes
```

We want to be able to download and upload lists of files to and from
these sshable hosts with almost no boilerplate code.

```python
import via_ssh

via_ssh.download_files(
    machine="lam",
    remote_src_local_dest_path_pairs=remote_src_local_dest_path_pairs
)
```
