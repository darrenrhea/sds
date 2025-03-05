from color_print_json import (
     color_print_json
)
import os
import requests

def lrlli_list_running_lambda_labs_instances_cli_tool():
    """
    As a warmup to starting instances at Lambda Labs,
    we will first list all the running instances.
    """
    api_key = os.environ.get("lambda_labs_api_key")
    url = "https://cloud.lambdalabs.com/api/v1/instances"
    headers = {
        "accept": "application/json"
    }

    # Perform the GET request with basic authentication (password left blank)
    response = requests.get(url, headers=headers, auth=(api_key, ""))

    # Check the response status and print the output
    if response.ok:
        print("Request was successful!")
        color_print_json(response.json())
    else:
        print("Error:", response.status_code)
        print(response.text)