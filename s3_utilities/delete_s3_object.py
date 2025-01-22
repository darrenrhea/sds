import subprocess


def delete_s3_object(s3_object_url: str):
    """
    Tries to delete an object from an S3 bucket.
    """
    assert isinstance(s3_object_url, str), f"s3_object_url should be a string, but you gave something of type {type(s3_object_url)=} namely {s3_object_url=}"
    assert s3_object_url.startswith('s3://'), f"ERROR: Expected {s3_object_url=} to start with 's3://' but you gave {s3_object_url=}"
    print(f"deleting:\n\n   {s3_object_url}")

    s3_object_url_without_prefix = s3_object_url[5:]
    bucket_name, key = s3_object_url_without_prefix.split('/', 1)

    args = [
        'aws',
        's3api',
        'delete-object',
        '--bucket',
        bucket_name,
        '--key',
        key
    ]
    completed_process = subprocess.run(
        args=args,
        capture_output=True,
    )
    if completed_process.returncode == 0:
        return True
    else:
        print(f"ERROR: we failed to delete {s3_object_url}")
        return False
