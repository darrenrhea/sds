from setuptools import setup

setup(
    name="hash_tools",
    version="0.1.0",
    py_modules=[
        "base64_to_hexidecimal",
        "blake2b_256_of_file",
        "faster_pseudo_sha256_of_file",
        "get_num_bytes_of_file",
        "git_sha1_of_file",
        "hash_tools",
        "hexidecimal_to_bytes",
        "is_lowercase_md5_checksum",
        "md5_of_file",
        "sha1_of_file",
        "sha256_of_file",
        "sha512_256_of_file",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
            "sha256base64 = sha256base64_cli_tool:sha256base64_cli_tool",
        ],
    },
)
