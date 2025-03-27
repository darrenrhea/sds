from setuptools import setup

setup(
    name="lambda_labs_utilities",
    version="0.1.0",
    py_modules=[
        "galoaalli_get_a_list_of_all_appropriate_lambda_labs_instances",
        "gllak_get_lambda_labs_api_key",
        "gpalli_get_parsed_available_lambda_labs_instances",
        "list_lambda_labs_instances",
        "list_running_lambda_labs_instances",
        "pgallgd_parse_god_awful_lambda_labs_gpu_description",
        "stlli_stop_these_lambda_labs_instances",
        "toalli_turn_on_a_lambda_labs_instance",
        "toalliott_turn_on_a_lambda_labs_instance_of_this_type",
        "translate_lambda_labs_names_to_instance_ids",
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
            "stlli_stop_these_lambda_labs_instances = stlli_stop_these_lambda_labs_instances_cli_tool:stlli_stop_these_lambda_labs_instances_cli_tool",
            "toalliftpn_turn_on_a_lambda_labs_instance_for_this_purpose_named = toalliftpn_turn_on_a_lambda_labs_instance_for_this_purpose_named_cli_tool:toalliftpn_turn_on_a_lambda_labs_instance_for_this_purpose_named_cli_tool",
        ],
    },
)
