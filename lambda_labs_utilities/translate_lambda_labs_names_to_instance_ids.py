import pprint
from list_lambda_labs_instances import (
     list_lambda_labs_instances
)

def translate_lambda_labs_names_to_instance_ids():
    """
    we have names like l0, l1, g0, g1, etc.
    We want to translate these to the instance_ids.
    """
    
    data = list_lambda_labs_instances()
    name_mapsto_instance_id = {}


    for instance in data:
        name = instance["name"]
        instance_id = instance["id"]
        name_mapsto_instance_id[name] = instance_id
    
    return name_mapsto_instance_id


if __name__ == "__main__":
    ans = translate_lambda_labs_names_to_instance_ids()
    pprint.pprint(ans)