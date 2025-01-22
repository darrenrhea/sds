from setuptools import setup

setup(
    name="via_ssh",
    version="0.1.0",
    py_modules=[
        "get_remote_home_directory_via_ssh",
        "raw_remote_directory_exists",
        "remote_directory_exists",
        "resolve_sshable_abbrev_to_username_hostname_port",
        "via_ssh",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "paramiko",
    ],
    entry_points={
        "console_scripts": [
        ],
    },
)
