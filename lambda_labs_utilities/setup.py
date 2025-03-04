from setuptools import setup

setup(
    name="lambda_labs_utilities",
    version="0.1.0",
    py_modules=[
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
            "lalli_list_available_lambda_labs_instances = lalli_list_available_lambda_labs_instances_cli_tool:lalli_list_available_lambda_labs_instances_cli_tool",
            "llli_list_lambda_labs_instances = llli_list_lambda_labs_instances_cli_tool:llli_list_lambda_labs_instances_cli_tool",
        ],
    },
)
