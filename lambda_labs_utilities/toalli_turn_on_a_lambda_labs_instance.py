import sys
from galoaalli_get_a_list_of_all_appropriate_lambda_labs_instances import (
     galoaalli_get_a_list_of_all_appropriate_lambda_labs_instances
)
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


def toalli_turn_on_a_lambda_labs_instance_of_this_type(
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


if __name__ == "__main__":
    available_instance_types = galoaalli_get_a_list_of_all_appropriate_lambda_labs_instances(
        min_gpu_memory_per_gpu_in_gigabytes=24,
        desired_num_gpus_per_instance=1,
        max_price_per_hour_in_cents=150,
        desired_instance_type_ids=["gpu_1x_a100_sxm4"],
    )
    if len(available_instance_types) == 0:
        print_red("No appropriate Lambda Labs instances available.")
        sys.exit(1)
    
    best_instance_type = available_instance_types[0]
    best_instance_type_id = best_instance_type["instance_type"]["instance_type_id"]
    best_region_id = best_instance_type["best_region_id"]
    print_yellow(f"Best instance type: {best_instance_type_id}")
    print_yellow(f"Best region: {best_region_id}")

    toalli_turn_on_a_lambda_labs_instance_of_this_type(
        instance_type_id=best_instance_type_id,
        region_id=best_region_id,
        what_to_call_the_instance="l0"
    )