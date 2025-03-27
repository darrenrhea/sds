from color_print_json import (
     color_print_json
)
from list_lambda_labs_instances import (
     list_lambda_labs_instances
)


def list_running_lambda_labs_instances():
    """
    List all the running instances at Lambda Labs,
    i.e. those that can be sshed into.
    """

    instances = list_lambda_labs_instances()
    
    
    running_instances = [
        instance
        for instance in instances
        if instance["status"] in ["active", "running"]
    ]

    return running_instances


if __name__ == "__main__":
    running_instances = list_running_lambda_labs_instances()
    color_print_json(running_instances)
    
      