import textwrap
from hash_tools import (
     base64_to_hexidecimal
)

import json
import subprocess



def maybe_get_sha256_of_s3_object(s3_object_url: str):
    """
    If the s3 object has a sha256 checksum already calculated
    for it in s3, this returns it.
    Otherwise, it returns None.
    See cs256_calculate_sha256_of_s3_object for a fix to
    the situation that a file does not have a sha256 checksum attached to it
    in s3.
    """
    assert s3_object_url.startswith('s3://')
    s3_object_url_without_prefix = s3_object_url[5:]
    bucket_name, key = s3_object_url_without_prefix.split('/', 1)

    args = [
        'aws',
        's3api',
        'get-object-attributes',
        '--bucket',
        bucket_name,
        '--key',
        key,
        '--object-attributes',
        'Checksum',
        '--output',
        'json'
    ]
    completed_process = subprocess.run(
        args=args,
        capture_output=True,
    )
    result = json.loads(
        completed_process.stdout.decode('utf-8')
    )
    if "Checksum" not in result:
        return None
    if "ChecksumSHA256" not in result["Checksum"]:
        return None
    sha256_in_base64 = result["Checksum"]["ChecksumSHA256"]
    sha256_in_hexidecimal = base64_to_hexidecimal(
        b64_str=sha256_in_base64
    )

    assert (
        len(sha256_in_hexidecimal) == 64
    ), textwrap.dedent(
        f"""\
        Ut-oh, something is up! SHA256 should be 64 characters long, but this said
        {sha256_in_hexidecimal=}
        which has length
        {len(sha256_in_hexidecimal)}
        """
    )

    return sha256_in_hexidecimal
