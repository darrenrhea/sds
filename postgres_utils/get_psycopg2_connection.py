"""
Gets credentials to make a postgres connection to a named database-server
"""
import os
import sys
import psycopg2
from colorama import Fore, Style



def get_psycopg2_connection(
    database_server_name: str,
    database_name: str
):
    """
    People can name the database-server that they want to connect to and this will give them
    the psycopg2 connection.
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

    print(
        dict(
            host=host,  # can this be zeus if it is defined in /etc/hosts?
            port=port,
            dbname=database_name,
            user=username,
            password=password,
        )
    )
    conn = psycopg2.connect(
        host=host,  # can this be zeus if it is defined in /etc/hosts?
        port=port,
        dbname=database_name,
        user=username,
        password=password,
    )
    return conn



