
import os


def get_num_bytes_of_file(file_path) -> int:
    num_bytes = os.stat(file_path).st_size
    return num_bytes
