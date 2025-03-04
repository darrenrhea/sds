from gpalli_get_parsed_available_lambda_labs_instances import (
     gpalli_get_parsed_available_lambda_labs_instances
)
from color_print_json import (
     color_print_json
)
import os
import requests


def lalli_list_available_lambda_labs_instances_cli_tool():
    """
    As a warmup to starting instances at Lambda Labs, we will first list all the instances.
    """
    jsonable = gpalli_get_parsed_available_lambda_labs_instances()
    # Check the response status and print the output
  
    color_print_json(jsonable)
  