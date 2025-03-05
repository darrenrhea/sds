from galoaalli_get_a_list_of_all_appropriate_lambda_labs_instances import (
     galoaalli_get_a_list_of_all_appropriate_lambda_labs_instances
)
from gpalli_get_parsed_available_lambda_labs_instances import (
     gpalli_get_parsed_available_lambda_labs_instances
)
from color_print_json import (
     color_print_json
)


def gaaallit_get_all_appropriate_available_lambda_labs_instance_types_cli_tool(
) -> None:
    galoaalli_get_a_list_of_all_appropriate_lambda_labs_instances(
        min_gpu_memory_per_gpu_in_gigabytes=24,
        desired_num_gpus_per_instance=1,
        max_price_per_hour_in_cents=200,
    )

  
    
  