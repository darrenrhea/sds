from setuptools import setup

setup(
    name="distributed_hash_table",
    version="0.1.0",
    py_modules=[
        "download_sha256_from_s3",
        "get_file_path_of_sha256",
        "store_file_by_sha256",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "colorama",
    ],
    entry_points={
        "console_scripts": [
            "get_file_path_of_sha256 = get_file_path_of_sha256_cli_tool:get_file_path_of_sha256_cli_tool",
            "store_file_by_sha256 = store_file_by_sha256_cli_tool:store_file_by_sha256_cli_tool",
        ],
    },
)
