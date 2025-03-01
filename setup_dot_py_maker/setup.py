from setuptools import setup

setup(
    name="setup_dot_py_maker",
    version="0.1.0",
    py_modules=[
        "ast_to_concrete_syntax"
        "msdp_make_setup_dot_py_cli_tool",
    ],
    license="MIT",
    long_description=("Tools for making setup.py automatically"),
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
            "msdp_make_setup_dot_py = msdp_make_setup_dot_py_cli_tool:msdp_make_setup_dot_py_cli_tool",
        ]
    }
)