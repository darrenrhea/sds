
from pathlib import Path
import better_json as bj


# A mapping from JSON field names to the typical SSH config directives
# (We’ll capitalize them properly when writing.)


def write_ssh_config_entry_lines(entry: dict):
    """
    Writes one block of the ~/.ssh/config file.
    """
    key_map = {
        "hostname": "HostName",
        "port": "Port",
        "user": "User",
        "identityfile": "IdentityFile",
        "forwardagent": "ForwardAgent",
        "stricthostkeychecking": "StrictHostKeyChecking",
        "userknownhostsfile": "UserKnownHostsFile",
        "usekeychain": "UseKeychain",
        "xauthlocation": "XAuthLocation",
        "localcommand": "LocalCommand",
        "sendenv": "SendEnv",
        "setenv": "SetEnv",
        "addkeystoagent": "AddKeysToAgent",
    }
    for k in key_map.keys():
        key_map[k] = key_map[k].lower()
    host = entry["host"]
    lines = []
    host_line = f"Host {host}"
    lines.append(host_line)

    # Write all known settings with proper indentation
    for k, v in entry.items():
        if k == "host":
            continue
        # Map to SSH config directive if we recognize k
        directive = key_map.get(k.lower(), k)  # fallback: use the key as-is if unknown
        # Indent by 4 spaces (as is common in SSH config)
        line = (f"    {directive} {v}")
        lines.append(line)

    # Blank line after each Host block for readability
    lines.append("")
    return lines

def write_ssh_config_lines(entries):
    """
    Writes the entire ~/.ssh/config file.
    """
    lines = []
    for entry in entries:
        entry_lines = write_ssh_config_entry_lines(entry)
        lines.extend(entry_lines)
    return lines


def write_ssh_config_content(entries):
    """
    Writes the entire ~/.ssh/config file as a string.
    """
    lines = write_ssh_config_lines(entries)
    return "\n".join(lines)

def write_ssh_config_file(
    entries: list[dict],
    out_file_path: Path,
):
    content = write_ssh_config_content(entries)
    out_file_path.write_text(content)


def test_write_ssh_config_file_1():
    input_json_path = Path("~/.ssh/manual_ssh_config.json").expanduser()
    out_file_path = Path("~/.ssh/generated_config").expanduser()
    entries = bj.load(input_json_path)
    content = write_ssh_config_content(entries)
    out_file_path.write_text(content)
    print(f"bat {out_file_path}")


if __name__ == "__main__":
    test_write_ssh_config_file_1()