import textwrap
from print_red import (
     print_red
)
import os
import sys


def gllak_get_lambda_labs_api_key() -> str:
    """
    Get the Lambda Labs API key from the unix / linux environment variable lambda_labs_api_key,
    or complain and exit if it is not set properly.
    """
    environment_variable_name = "lambda_labs_api_key"
    if environment_variable_name not in os.environ:
        print_red(
            textwrap.dedent(
                f"""\
                Please set the environment variable {environment_variable_name}
                to a Lambda Labs API key.  You can create an API at
                https://cloud.lambdalabs.com/api-keys
                """
            )
        )
        sys.exit(1)
    api_key = os.environ.get("lambda_labs_api_key")
    assert (
        api_key.startswith("secret_")
    ), "The Lambda Labs API key should begin with secret_"
    
    return api_key

   