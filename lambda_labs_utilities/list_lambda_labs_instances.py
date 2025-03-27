import sys
from print_red import print_red
import requests
from gllak_get_lambda_labs_api_key import (
     gllak_get_lambda_labs_api_key
)


def list_lambda_labs_instances():
    """
    List all the instances at Lambda Labs.
    """
    api_key = gllak_get_lambda_labs_api_key()
    url = "https://cloud.lambdalabs.com/api/v1/instances"
    headers = {
        "accept": "application/json"
    }

    # Perform the GET request with basic authentication (password left blank)
    response = requests.get(url, headers=headers, auth=(api_key, ""))

    # Check the response status and print the output
    if response.ok:
        obj = response.json()
    else:
        print_red("Error:", response.status_code)
        print_red(response.text)
        sys.exit(1)
    
    data = obj["data"]
    
    return data
      