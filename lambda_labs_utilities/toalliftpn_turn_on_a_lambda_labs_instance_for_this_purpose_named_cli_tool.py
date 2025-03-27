import argparse

from toalli_turn_on_a_lambda_labs_instance import (
     turn_on_a_lambda_labs_instance
)


def toalliftpn_turn_on_a_lambda_labs_instance_for_this_purpose_named_cli_tool():
    argp = argparse.ArgumentParser()
    
    argp.add_argument(
        "--purpose",
        required=True,
        help="The purpose of the instance, like fake_data_generation or training",
    )
    
    argp.add_argument(
        "--name",
        required=True,
        help="give a name to the instance, like l0",
    )
   
    opt = argp.parse_args()
    what_to_call_the_instance = opt.name
    purpose = opt.purpose
    turn_on_a_lambda_labs_instance(
        what_to_call_the_instance=what_to_call_the_instance,
        purpose=purpose,
    )
