from pgallgd_parse_god_awful_lambda_labs_gpu_description import (
     pgallgd_parse_god_awful_lambda_labs_gpu_description
)
import os
import requests


def gpalli_get_parsed_available_lambda_labs_instances():
    """
    In order to automatically launch a lambda_labs instance, we need to
    get a list of available instance types.
    Unfortunately, the response from lambda_labs contains free-form strings
    describing the GPUs available on each instance type,
    so we have to parse those.
    """
    api_key = os.environ.get("lambda_labs_api_key")
    url = "https://cloud.lambdalabs.com/api/v1/instance-types"
    headers = {
        "accept": "application/json"
    }

    # Perform the GET request with basic authentication (password left blank)
    response = requests.get(url, headers=headers, auth=(api_key, ""))
    
    # Check the response status and print the output
    if response.ok:
        unparsed_list_of_available = response.json()
        assert "data" in unparsed_list_of_available
        instancetypename_to_desc = unparsed_list_of_available["data"]
        parsed = []
        for informal_name, desc in instancetypename_to_desc.items():
            instance_type = desc["instance_type"]
            gpu_description_str = instance_type["gpu_description"]
            
            gpu_memory_gb = pgallgd_parse_god_awful_lambda_labs_gpu_description(
                gpu_description_str
            )
            
            instance_type["gpu_memory_gb"] = gpu_memory_gb
            instance_type["instance_type_id"] = informal_name
            parsed.append(desc)
        return parsed


    else:
        print("Error:", response.status_code)
        print(response.text)