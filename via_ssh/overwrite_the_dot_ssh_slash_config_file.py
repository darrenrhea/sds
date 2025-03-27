from pathlib import Path
from write_ssh_config_file import write_ssh_config_file
import better_json as bj

def write_dot_ssh_config_file_from_jsonables(
    manual_entries: list[dict],
    generated_entries: list[dict],
    out_file_path: Path,
):
    # check that no host name is in both:
    manual_host_names = {entry["host"] for entry in manual_entries}
    generated_host_names = {entry["host"] for entry in generated_entries}
    if manual_host_names & generated_host_names:
        raise ValueError("Some host names are in both manual and generated entries")
    
    entries = manual_entries + generated_entries

    write_ssh_config_file(
        entries=entries,
        out_file_path=out_file_path,
    )


def overwrite_the_dot_ssh_slash_config_file():
    """
    This combines the manual and generated SSH config entries into a single file.
    """
    manual_entries = bj.load(Path("~/.ssh/manual_ssh_config.json5").expanduser())
    generated_entries = bj.load(Path("~/.ssh/dynamic_ssh_config.json5").expanduser())
    out_file_path = Path("~/.ssh/config").expanduser()
    
    write_dot_ssh_config_file_from_jsonables(
        manual_entries=manual_entries,
        generated_entries=generated_entries,
        out_file_path=out_file_path,
    )

if __name__ == "__main__":
    overwrite_the_dot_ssh_slash_config_file()
    print(f"bat {Path('~/.ssh/config').expanduser()}")