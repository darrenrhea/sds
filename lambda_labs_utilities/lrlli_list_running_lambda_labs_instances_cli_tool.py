import sys
import textwrap
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
        obj = response.json()
        color_print_json(obj)
    else:
        print("Error:", response.status_code)
        print(response.text)
        sys.exit(1)
    
    data = obj["data"]
    
    running_instances = [
        instance
        for instance in data
        if instance["status"] in ["active", "running"]
    ]

    print("You might want to add the following to your ~/.ssh/config file:")
    for running_instance in running_instances:
        ip = running_instance["ip"]
        print(
            textwrap.dedent(
                f"""
                Host l0
                    forwardagent yes
                    user ubuntu
                    port 22
                    hostname {ip}
                    identityfile ~/.ssh/id_rsa
                """
            )
        )
      