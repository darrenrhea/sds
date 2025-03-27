from pathlib import Path
from typing import Optional
from print_yellow import print_yellow
from save_jsonable import (
     save_jsonable
)
import os

from color_print_json import color_print_json

def parse_ssh_config_lines(lines):
    """
    Parses the SSH config file and returns a dictionary where each key is a host pattern
    and its value is a dictionary of configuration options.
    """

    entries = []
    host = None
    entry = None

    for line in lines:
        # Remove leading/trailing whitespace
        line = line.strip()
        # Skip blank lines and comments
        if not line or line.startswith('#'):
            continue

        # Split line into keyword and value (separated by any whitespace)
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue  # skip malformed lines

        key, value = parts
        key_lower = key.lower()

        if key_lower == 'host':
            if entry is not None:
                entries.append(entry)
            # When encountering a new Host line, split in case multiple hosts are defined
            host = value
            print_yellow(f"host: {host}")
            entry = {
                'host': host,
            }
        else:
            assert host is not None, "Config line found before Host line"
            # For all configuration lines, add the option to every host in the current block.
            entry[key_lower] = value
    
    if entry is not None:
        entries.append(entry)
    
    return entries


def parse_ssh_config_file(
    file_path: Optional[Path] = None
):
    """
    Parses the SSH config file and returns a dictionary where each key is a host pattern
    and its value is a dictionary of configuration options.
    """
    if file_path is None:
        file_path = Path("~/.ssh/config").expanduser()
    
    if not file_path.resolve().is_file():
        raise FileNotFoundError(f"SSH config file not found at {file_path}")
    
    lines = file_path.read_text().splitlines()
    return parse_ssh_config_lines(lines)

if __name__ == '__main__':
    try:
        ssh_config_entries = parse_ssh_config_file()
        for entry in ssh_config_entries:
            host = entry['host']
            print(f"Host: {host}")
            for option, value in entry.items():
                if option == 'host':
                    continue
                print(f"  {option}: {value}")
            print()
    except Exception as e:
        print(f"Error: {e}")
    
    color_print_json(ssh_config_entries)
    out_dir = Path("~/.ssh").expanduser()
    out_dir.mkdir(exist_ok=True)
    out_file_path = out_dir / 'manual_ssh_config.json'
    save_jsonable(
        fp=out_file_path,
        obj=ssh_config_entries
    )
    print(f"Saved ssh_config.json to {out_file_path}")