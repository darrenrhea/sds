from pathlib import Path
import better_json as bj


def load_map_from_resolved_abs_file_path_to_sha256(
    json_file_path: Path
):
    assert json_file_path.is_file()
    return bj.load(json_file_path)
