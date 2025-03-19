from typing import List
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


def stlli_stop_these_lambda_labs_instances(
    instance_ids: List[str],
) -> None:
    """
    Terminates the given Lambda Labs instances.
    """
    api_key = gllak_get_lambda_labs_api_key()
  
    url = "https://cloud.lambdalabs.com/api/v1/instance-operations/terminate"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # if you leave out the image, it will choose the default image
    json_to_post = {
        "instance_ids": instance_ids,
    }
    response = requests.post(
        url,
        json=json_to_post,
        headers=headers,
        auth=(api_key, "")
    )
    if response.ok:
        print_yellow("Stopping theses Lambda Labs instances:")
        color_print_json(response.json())
    else:
        print_red("Error terminating Lambda Labs instances:")
        print(f"{response.status_code=}")
        print_red(response.text)


if __name__ == "__main__":
    instance_ids = [
        "8cdec69999b343caadfb625a23430b64",
    ]
    stlli_stop_these_lambda_labs_instances(
        instance_ids=instance_ids,
    )
        