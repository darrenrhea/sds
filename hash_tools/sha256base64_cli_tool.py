
from base64 import b64encode
from pathlib import Path
import sys

from hash_tools import sha256_of_file
from hexidecimal_to_bytes import hexidecimal_to_bytes


def sha256base64_cli_tool():
    """
    Because AWS s3 tends to use base64 to encode sha256 checksums.
    """
    file_path = Path(sys.argv[1]).resolve()
    
    hex_str = sha256_of_file(
        file_path=file_path
    )
    asbytes = hexidecimal_to_bytes(hex_str)
    b64_str = b64encode(asbytes)
    print(b64_str.decode("ascii"))
  