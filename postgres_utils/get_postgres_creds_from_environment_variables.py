import os
import sys
from colorama import Fore, Style


def get_postgres_creds_from_environment_variables(
    database_server_name: str,
    database_name: str
):
    """
    Gets credentials to make a postgres connection to a named database-server
    from environment variables.
    """
    try:
        host_key = f"{database_server_name}_POSTGRES_HOSTNAME"
        host = os.environ[host_key]
    except KeyError:
        print(f"{Fore.RED}You need to define the environment variable {host_key}, something like 45.56.121.210{Style.RESET_ALL}")
        sys.exit(1)

    try:
        port_key = f"{database_server_name}_POSTGRES_PORT"
        port = os.environ[port_key]
    except KeyError:
        print(f"{Fore.RED}You need to define the environment variable {port_key}, usually 5432{Style.RESET_ALL}")
        sys.exit(1)

    try:
        username_key = f"{database_server_name}_POSTGRES_USERNAME"
        username = os.environ[username_key]
    except KeyError:
        print(f"{Fore.RED}You need to define the environment variable {username_key}{Style.RESET_ALL}")
        sys.exit(1)

    try:
        password_key = f"{database_server_name}_POSTGRES_PASSWORD"
        password = os.environ[password_key]
    except KeyError:
        print(f"{Fore.RED}You need to define the environment variable {password_key}{Style.RESET_ALL}, the password")
        sys.exit(1)

    
    ans = dict(
        host=host,
        port=port,
        dbname=database_name,
        user=username,
        password=password,
    )

    return ans




