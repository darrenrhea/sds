import textwrap
from what_os_is_this import (
     what_os_is_this
)
from print_to_stdout_without_linefeed import (
     print_to_stdout_without_linefeed
)
from print_yellow import (
     print_yellow
)
from print_green import (
     print_green
)
from store_file_by_sha256 import (
    store_file_by_sha256
)
import argparse
from pathlib import Path


def store_file_by_sha256_cli_tool():
    argp = argparse.ArgumentParser(
        description=textwrap.dedent(
            """
            Given the path to a file, store_file_by_sha256 stores it "under"
            its sha256 hash in s3.
            As a convenience, it keeps the extension.
            """
        ),
        usage=textwrap.dedent(
            """
            # save the sha256 hash of the file to another file:
            store_file_by_sha256 ~/Downloads/GLgqrN_XwAAVgk9.jpg > the_sha256_hash
            
            # look at that image:
            pri $(cat the_sha256_hash)

            # using environment variables:
            the_sha256_hash=$(store_file_by_sha256 ~/Downloads/GLgqrN_XwAAVgk9.jpg)
            pri $the_sha256_hash
            """
        ),
    )
    argp.add_argument("file_path", type=Path)
    args = argp.parse_args()
    sha256_hash = store_file_by_sha256(args.file_path)
    print_green(f"Stored {args.file_path} as its sha256 hash")
    if what_os_is_this() == "macos":
        import pyperclip
        pyperclip.copy(sha256_hash)
        print_green("Note that the hash is now on the clipboard if you want to paste it")
    else:
        print_yellow(
            "You are on linux, which oftentimes means you sshed in, so we don't put it on the clipboard, but just to stdout"
        )
    # regardless of the operating system, we print the hash to stdout:
    print_to_stdout_without_linefeed(sha256_hash)

      

