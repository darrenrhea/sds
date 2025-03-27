from toalliott_turn_on_a_lambda_labs_instance_of_this_type import (
     toalliott_turn_on_a_lambda_labs_instance_of_this_type
)
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


def turn_on_a_lambda_labs_instance(
    what_to_call_the_instance: str,
    purpose: str,
):
    valid_purposes = ["fake_data_generation", "training",]

    assert (
        purpose in valid_purposes
    ), f"Unknown purpose: {purpose} is not a valid purpose. Possibities are {valid_purposes})"

    if purpose == "fake_data_generation":
        min_gpu_memory_per_gpu_in_gigabytes=0
        desired_num_gpus_per_instance=1
        max_price_per_hour_in_cents=150
        desired_instance_type_ids=[
            "gpu_1x_a10",
            "gpu_1x_a100_sxm4",
            "gpu_1x_rtx6000",
        ]
    elif purpose == "training":
        min_gpu_memory_per_gpu_in_gigabytes=24
        desired_num_gpus_per_instance=1
        max_price_per_hour_in_cents=150
        desired_instance_type_ids=["gpu_1x_a100_sxm4"]
    else:
        raise NotImplementedError()
 
    available_instance_types = (
        galoaalli_get_a_list_of_all_appropriate_lambda_labs_instances(
            min_gpu_memory_per_gpu_in_gigabytes=min_gpu_memory_per_gpu_in_gigabytes,
            desired_num_gpus_per_instance=desired_num_gpus_per_instance,
            max_price_per_hour_in_cents=max_price_per_hour_in_cents,
            desired_instance_type_ids=desired_instance_type_ids,
        )
    )
    if len(available_instance_types) == 0:
        print_red("No appropriate Lambda Labs instances available.")
        sys.exit(1)
    
    best_instance_type = available_instance_types[0]
    best_instance_type_id = best_instance_type["instance_type"]["instance_type_id"]
    best_region_id = best_instance_type["best_region_id"]
    print_yellow(f"Best instance type: {best_instance_type_id}")
    print_yellow(f"Best region: {best_region_id}")

    toalliott_turn_on_a_lambda_labs_instance_of_this_type(
        instance_type_id=best_instance_type_id,
        region_id=best_region_id,
        what_to_call_the_instance=what_to_call_the_instance,
    )