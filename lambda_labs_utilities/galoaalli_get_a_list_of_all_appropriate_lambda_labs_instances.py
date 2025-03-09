from gpalli_get_parsed_available_lambda_labs_instances import (
     gpalli_get_parsed_available_lambda_labs_instances
)
from color_print_json import (
     color_print_json
)


def galoaalli_get_a_list_of_all_appropriate_lambda_labs_instances(
    min_gpu_memory_per_gpu_in_gigabytes=24,
    desired_num_gpus_per_instance=1,
    max_price_per_hour_in_cents=150,
    desired_instance_type_ids=None,
) -> None:
    """
    We need to determine which, of all the available Lambda Labs instances,
    are appropriate for our needs. We will filter based on the following criteria:
    - min_gpu_memory_per_gpu_in_gigabytes
    - desired_num_gpus_per_instance
    - max_price_per_hour_in_cents
    - forbidden_instance_type_ids
    and select a closest region for each instance type.
    Then sort by price.
    """
    all_types = gpalli_get_parsed_available_lambda_labs_instances()

    forbidden_instance_type_ids = [
        "gpu_1x_gh200"
    ]

    filtered_prior_to_avaiability = []

    for x in all_types:

        instance_type = x["instance_type"]
        instance_type_id = instance_type["instance_type_id"]
        print("Considering instance type:", instance_type_id)
        gpu_memory_gb_for_each_gpu = instance_type["gpu_memory_gb"]
        price_cents_per_hour = instance_type["price_cents_per_hour"]
        specs = instance_type["specs"]
        num_gpus_per_instance = specs["gpus"]
        print(num_gpus_per_instance)
        good = True
        has_at_least_one_gpu = gpu_memory_gb_for_each_gpu is not None
        if not has_at_least_one_gpu:
            good = False
        if gpu_memory_gb_for_each_gpu is not None and gpu_memory_gb_for_each_gpu < min_gpu_memory_per_gpu_in_gigabytes:
            good = False
        if num_gpus_per_instance != desired_num_gpus_per_instance:
            good = False
        if max_price_per_hour_in_cents is not None and price_cents_per_hour > max_price_per_hour_in_cents:
            good = False
        if instance_type_id in forbidden_instance_type_ids:
            good = False
        if desired_instance_type_ids is not None and instance_type_id not in desired_instance_type_ids:
            good = False
        
        if good: 
            filtered_prior_to_avaiability.append(x)
        
    close_enough_regions_in_order_of_most_desirable_to_least_desirable = [
        "us-east-1",
        "us-east-2",
        "us-east-3",
        "us-west-1",
        "us-west-2",
        "us-midwest-1",
        "us-south-1",
        "us-south-2",
    ]
    too_far = [
        "asia-south-1"
    ]
    
    color_print_json(all_types)

    geographicly_close_enough_types_with_region = []
    for x in filtered_prior_to_avaiability:
        instance_type = x["instance_type"]
        regions_with_capacity_available = x["regions_with_capacity_available"]
        region_ids_with_capacity_available = [z["name"] for z in regions_with_capacity_available]
        instance_type_id = instance_type["instance_type_id"]
        print("Considering instance type:", instance_type_id)
        print(f"{regions_with_capacity_available=}")
        close_enough = False
        best_region_id = None
        for region_id in close_enough_regions_in_order_of_most_desirable_to_least_desirable:
            if region_id in region_ids_with_capacity_available:
                close_enough = True
                best_region_id = region_id
                break
        if close_enough:
            x["best_region_id"] = best_region_id
            geographicly_close_enough_types_with_region.append(
                x
            )
    sorted_by_price = sorted(
        geographicly_close_enough_types_with_region,
        key=lambda x: x["instance_type"]["price_cents_per_hour"],
        reverse=False
    )
    print("Sorted")
    color_print_json(sorted_by_price)
    return geographicly_close_enough_types_with_region
  
    
  