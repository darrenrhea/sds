"""
No wonder people use actual databases.
No sane file locking available.
rm test.jsonl && seq 1 10 | parallel 'python test_append_one_object_to_jsonlines_file.py {}'
"""
from better_json import append_one_object_to_jsonlines_file
import sys
from pathlib import Path

tag = int(sys.argv[1])

jsonlines_path = Path("test.jsonl")


for index in range(10000):
    obj = { "dog": f"{(10*index + tag):06d}", "tag": tag}
    append_one_object_to_jsonlines_file(
        obj=obj,
        jsonlines_path=jsonlines_path
    )