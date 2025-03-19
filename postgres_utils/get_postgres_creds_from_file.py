from pathlib import Path
import pprint
import better_json as bj


def get_postgres_creds_from_file():
    """
    Gets credentials to make a postgres connection.
    """
    file_path = Path(
        "~/.othercreds/postgres_creds.json"
    ).expanduser().resolve()

    assert file_path.exists(), f"{file_path} does not exist"
    
    obj = bj.load(file_path)
    hostname = obj["hostname"]
    port = obj["port"]
    database_name = obj["database_name"]
    username = obj["username"]
    password = obj["password"]

    ans = dict(
        host=hostname,
        port=port,
        dbname=database_name,
        user=username,
        password=password,
    )

    return ans


if __name__ == "__main__":
    print(get_postgres_creds_from_file())