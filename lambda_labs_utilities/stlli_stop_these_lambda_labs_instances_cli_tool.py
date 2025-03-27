from stlli_stop_these_lambda_labs_instances import (
     stlli_stop_these_lambda_labs_instances
)
import argparse
import textwrap
from print_red import (
     print_red
)
from translate_lambda_labs_names_to_instance_ids import (
     translate_lambda_labs_names_to_instance_ids
)


def stlli_stop_these_lambda_labs_instances_cli_tool():
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "names",
        nargs="+",
        help="The names of the Lambda Labs instances to stop."
    )
    opt = argp.parse_args()
    names = opt.names

    instance_ids = []
    name_to_instance_id = translate_lambda_labs_names_to_instance_ids()
    
    for name in names:
        instance_id = name_to_instance_id.get(name)
        if instance_id is None:
            print_red(f"Could not resolve the instance named {name} to an instance_id")
            continue
        instance_ids.append(instance_id)
    

    stlli_stop_these_lambda_labs_instances(
        instance_ids=instance_ids,
    )

    print(
        textwrap.dedent(
            """"
            you can run lrlli_list_running_lambda_labs_instances"
            or visit:"
            
            https://cloud.lambdalabs.com/login
            """
        )
    )
        