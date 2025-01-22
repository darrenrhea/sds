from ls_with_glob_for_s3 import (
     ls_with_glob_for_s3
)
import argparse
import sys
import textwrap


def s3ls_cli_tool():
    """
    The aws s3 ls command is shockingly bad,
    for instance the absence of wildcard support is a big problem.

    https://stackoverflow.com/questions/39857802/check-if-file-exists-in-s3-using-ls-and-wildcard
    
    We want to find all files inside a s3 pseudo-folder with a glob wildcard, like:

    s3ls s3://awecomai-original-videos/ingested/BOSvDAL_PGM_core_esp_06-09-2024 *lq*
    
    """

    usage_str = textwrap.dedent(
        """\
        Usage: s3ls <s3_directory_path> <glob_pattern_in_quotations_so_that_the_shell_doesnt_expand_it>"
        
        s3ls s3://awecomai-original-videos '*_lq*'
        s3ls s3://awecomai-original-videos/ '*_lq*'
        s3ls awecomai-original-videos '*_lq*'
        s3ls awecomai-original-videos/ '*_lq*'

        s3ls -r s3://awecomai-original-videos '*_lq*'
        s3ls -r s3://awecomai-original-videos '*_lq*'
        """
    )
    if not (
        len(sys.argv) == 3
        or
        (
            len(sys.argv) == 4
            and
            sys.argv[1] in ["--recursive", "-r"]
        )
    ):
        print(usage_str)
        sys.exit(1)

    argp = argparse.ArgumentParser(
        description="List all files in an s3 pseudo-folder that match a wildcard glob pattern",
        usage=usage_str
    )

    argp.add_argument('s3path', type=str, help='the s3 pseudofolder to look through')
    
    argp.add_argument('wildcard', type=str, help='the wildcard pattern to match against')
    argp.add_argument('--recursive', '-r', action='store_true', help='whether to search recursively')
    args = argp.parse_args()
    s3_pseudo_folder = args.s3path
    glob_pattern = args.wildcard
    recursive = args.recursive
    
    list_of_strings = ls_with_glob_for_s3(
        s3_pseudo_folder=s3_pseudo_folder,
        glob_pattern=glob_pattern,
        recursive=recursive
    )
    
    for s in list_of_strings:
        print(s)

