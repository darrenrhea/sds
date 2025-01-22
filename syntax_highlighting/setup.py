from setuptools import setup

setup(
    name="syntax_highlighting",
    version="0.1.0",
    py_modules=[
        "list_all_pygments_styles",
        "print_syntax_highlighted_json",
        "print_syntax_highlighted_python",
        "print_syntax_highlighted_yaml",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
            "bt = bt_cli_tool:bt_cli_tool",
        ],
    },
)
