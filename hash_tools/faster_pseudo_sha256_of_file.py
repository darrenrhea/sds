from get_num_bytes_of_file import (
     get_num_bytes_of_file
)
import os
import hashlib


def faster_pseudo_sha256_of_file(file_path):
    """
    Some of the files we deal with are enormous enough (like 400 gigabytes)
    that sha256_ing them
    would take an hour.
    Instead, we do hash up to the first 128 MB,
    then the last 128 MB of the file.
    then the first 4096 of every Megabyte.
    """

    file_length = get_num_bytes_of_file(file_path=file_path)
    chunk_size = 4096
    prefix_length = 2**27
    suffix_length = 2**27
    m = hashlib.sha256()
    with open(os.path.expanduser(file_path), "rb") as fptr:
        prefix_bytes_processed = 0
        for chunk in iter(lambda: fptr.read(chunk_size), b""):
            m.update(chunk)
            prefix_bytes_processed += len(chunk)
            if prefix_bytes_processed >= prefix_length:
                break
        assert prefix_bytes_processed == min(
            prefix_length, file_length
        ), "Ut-oh, something is up!"
        fptr.seek(
            -min(file_length, suffix_length),  # how many bytes before the end
            2,  # 2 is a code that means "starting-from-the-end-of-the-file"
        )
        suffix_bytes_processed = 0
        for chunk in iter(lambda: fptr.read(chunk_size), b""):
            # print(f"suffix chunk {chunk}")
            m.update(chunk)
            suffix_bytes_processed += len(chunk)
            if suffix_bytes_processed >= suffix_length:
                break
        assert suffix_bytes_processed == min(
            suffix_length, file_length
        ), f"Ut-oh, went over somehow? {suffix_bytes_processed=}"

        # finally, the first 4096 bytes of every megabyte of the file:
        whole_number_of_megabytes = file_length // (2**20)
        num_megabytes_processed = 0
        for k in range(whole_number_of_megabytes):
            fptr.seek(
                k * (2**20),  # how many bytes offset from the beginning
                0,  # 0 is a code that means "starting-from-the-beginning-of-the-file"
            )
            fptr.read(chunk_size)
            m.update(chunk)
            num_megabytes_processed += 1
            if num_megabytes_processed % 1000 == 0:
                print(f"{num_megabytes_processed//1000} GBs processed")
        assert num_megabytes_processed == whole_number_of_megabytes

    return m.hexdigest()

