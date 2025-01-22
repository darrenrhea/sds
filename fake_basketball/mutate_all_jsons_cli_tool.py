"""
We forgot league in the referees json.
Luckily they are all the same league euroleague.
"""

import better_json as bj
from pathlib import Path


def mutate_all_jsons_cli_tool():
    folder = Path.cwd()
    for json_path in folder.glob("*.json"):
        print(json_path)
        data = bj.load(json_path)
        data["league"] = "euroleague"
        bj.dump(
            obj=data,
            fp=json_path
        )
