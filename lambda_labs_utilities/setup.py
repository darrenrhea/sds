from setuptools import setup

setup(
    name="lambda_labs_utilities",
    version="0.1.0",
    py_modules=[
        "gllak_get_lambda_labs_api_key",
        "gpalli_get_parsed_available_lambda_labs_instances",
        "pgallgd_parse_god_awful_lambda_labs_gpu_description",
        "toalli_turn_on_a_lambda_labs_instance",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
            "gaaallit_get_all_appropriate_available_lambda_labs_instance_types = gaaallit_get_all_appropriate_available_lambda_labs_instance_types_cli_tool:gaaallit_get_all_appropriate_available_lambda_labs_instance_types_cli_tool",
            "lalli_list_available_lambda_labs_instances = lalli_list_available_lambda_labs_instances_cli_tool:lalli_list_available_lambda_labs_instances_cli_tool",
            "lrlli_list_running_lambda_labs_instances = lrlli_list_running_lambda_labs_instances_cli_tool:lrlli_list_running_lambda_labs_instances_cli_tool",
        ],
    },
)
