from setuptools import setup

setup(
    name="s3_utilities",
    version="0.1.0",
    py_modules=[
        "copy_s3_object_url_to_this_destination_with_sha256_checksum",
        "could_be_an_s3_file_uri",
        "cs256_calculate_sha256_of_s3_object",
        "delete_s3_object",
        "download_this_s3_file_uri_to_this_file_path",
        "get_bucket_name_and_s3_key_from_s3_file_uri",
        "is_good_bucket_name",
        "list_all_s3_keys_in_this_bucket_with_this_prefix",
        "maybe_get_sha256_of_s3_object",
        "upload_file_path_to_s3_file_uri",
        "upload_to_s3_cli",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
            "download_from_s3 = download_from_s3_cli_tool:download_from_s3_cli_tool",
            "s3ls = s3ls_cli_tool:s3ls_cli_tool",
            "s3sha256 = s3sha256_cli_tool:s3sha256_cli_tool",
        ],
    },
)
