from color_print_json import color_print_json
from pathlib import Path
import textwrap
from list_lambda_labs_instances import (
     list_lambda_labs_instances
)
import better_json as bj
from overwrite_the_dot_ssh_slash_config_file import (
     overwrite_the_dot_ssh_slash_config_file
)
from print_green import print_green



def lrlli_list_running_lambda_labs_instances_cli_tool():
    """
    List all the instances at Lambda Labs.
    If they are running, add them to the ~/.ssh/config file.
    """
    
    data = list_lambda_labs_instances()
    
    running_instances = [
        instance
        for instance in data
        if instance["status"] in ["active", "running"]
    ]
    
    other_instances = [
        instance
        for instance in data
        if instance["status"] not in ["active", "running"]
    ]
    if len(other_instances) > 0:
        print("The following instances are not yet running:")
        color_print_json(other_instances)

    if len(running_instances) == 0:
        print("No running instances found.")
    else:
        print_green(
            textwrap.dedent(
                """
                We are adding the following entries
                to your ~/.ssh/config file
                that you can ssh into the instances easily.
                """
            )
        )
        entries = []
        for running_instance in running_instances:
            ip = running_instance["ip"]
            name = running_instance.get("name", "unknownname")
            print(
                textwrap.dedent(
                    f"""
                    Host {name}
                        forwardagent yes
                        user ubuntu
                        port 22
                        hostname {ip}
                        identityfile ~/.ssh/id_rsa
                    """
                )
            )
            entry = {
                "host": name,
                "forwardagent": "yes",
                "user": "ubuntu",
                "port": "22",
                "hostname": ip,
                "identityfile": "~/.ssh/id_rsa",
                "StrictHostKeyChecking": "accept-new",
            }
            entries.append(entry)
        
        generated_file_path = Path("~/.ssh/dynamic_ssh_config.json5").expanduser()
        bj.dump(obj=entries, fp=generated_file_path)
        overwrite_the_dot_ssh_slash_config_file()
        
        