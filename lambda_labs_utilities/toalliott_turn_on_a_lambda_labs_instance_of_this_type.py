from print_red import (
     print_red
)
from print_yellow import (
     print_yellow
)
from color_print_json import (
     color_print_json
)
from gllak_get_lambda_labs_api_key import (
     gllak_get_lambda_labs_api_key
)
import requests


def toalliott_turn_on_a_lambda_labs_instance_of_this_type(
    instance_type_id: str,
    region_id: str,
    what_to_call_the_instance: str,
) -> None:
    """
    As a warmup to starting instances at Lambda Labs,
    we will first list all the running instances.

    curl -u ${lambda_labs_api_key}: -X GET "https://cloud.lambdalabs.com/api/v1/images" -H 'accept: application/json'
    """
    api_key = gllak_get_lambda_labs_api_key()
  
    url = "https://cloud.lambdalabs.com/api/v1/instance-operations/launch"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # if you leave out the image, it will choose the default image
    json_to_post = {
        "region_name": region_id,
        "instance_type_name": instance_type_id,
        "ssh_key_names":["aang"],
        "file_system_names":[],
        "name": what_to_call_the_instance,
        # "image":{"id":"string"},
        "user_data":""
    }
    response = requests.post(
        url,
        json=json_to_post,
        headers=headers,
        auth=(api_key, "")
    )
    if response.ok:
        print_yellow(f"Instance of type {instance_type_id} starting:")
        color_print_json(response.json())
    else:
        print_red("Error starting Lambda Labs instance:")
        print(f"{response.status_code=}")
        print_red(response.text)
