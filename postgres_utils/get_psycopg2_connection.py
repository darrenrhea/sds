from get_postgres_creds_from_file import (
     get_postgres_creds_from_file
)
import psycopg2


def get_psycopg2_connection():
    """
    Grabs the credentials from file and returns the psycopg2 connection.
    """
    
    creds = get_postgres_creds_from_file()

    conn = psycopg2.connect(
        host=creds["host"],
        port=creds["port"],
        dbname=creds["dbname"],
        user=creds["user"],
        password=creds["password"],
    )

    return conn



