"""
This takes in a set of video frames, i.e. a set_of_clip_id_frame_index_pairs.
It also takes in a set of models, i.e. final_model_ids.
It then infers the frames using the specified models.
Then it publishes the results to a web interface.
"""

import sys
from print_green import (
     print_green
)
from get_clip_id_and_frame_index_from_original_file_name import (
     get_clip_id_and_frame_index_from_original_file_name
)
from pathlib import Path
from collections import defaultdict
import pprint
import better_json as bj

def hd71_sort_as_this_list_is_sorted(
    list_defining_the_sort_order,
    list_to_sort
):
    """
    Return a new list containing the elements of `list_to_sort`
    sorted in the same order as they appear in `list_defining_the_sort_order`.

    Both lists must contain hashable elements, and every element
    of list_to_sort must be in list_defining_the_sort_order.
    """
    # Build a lookup of each item’s position in the big list
    order = {item: idx for idx, item in enumerate(list_defining_the_sort_order)}
    # Sort the smaller list by that position
    return sorted(list_to_sort, key=lambda x: order[x])


models_in_order = [
    # "summerleague2025allnbafloorrev4epoch1",
    # "summerleague2025allnbafloorrev5epoch1",
    # "summerleague2025allnbafloorrev5epoch2",
    # "summerleague2025allnbafloorrev5epoch3",

    "summerleague2025batchsize2epoch340",
    "summerleague2025restart1epoch300",
    "summerleague2025restart2epoch280",

    # "summerleague2025allnbafloorrev5epoch4",
    # "summerleague2025allnbafloorrev5epoch5",
    # "summerleague2025allnbafloorrev4epoch4",
    # "summerleague2025floorrev3epoch100",
    # "summerleague2025floorrev1epoch10",
    # "summerleague2025floorrev2epoch113",
    # "summerleague2025allnbafloorrev3epoch20",
    # "summerleague2025allnbafloorrev4epoch2",
    # "summerleague2025allnbafloorrev4epoch5",
    # "summerleague2025allnbafloorrev4epoch29",
    # # "summerleague2025floorrev2epoch7",
    # # "summerleague2025floorrev2epoch56",
    # "summerleague2025floorrev3epoch65",
]
# every image has at at least one model?
# eventually, an indicator matrix of whether it is inferred or not.
rows = []
set_of_clip_id_frame_index_pairs = set()
obj = bj.load("bad_frames.json5")
for clip_id, bad_frame_descs in obj.items():
    for bad_frame_desc in bad_frame_descs:
        frame_index = bad_frame_desc[0]
        set_of_clip_id_frame_index_pairs.add((clip_id, frame_index))

print_green("Set of clip_id and frame_index pairs:")
pprint.pprint(set_of_clip_id_frame_index_pairs)
for model_id in models_in_order:
    print_green(f"Model ID: {model_id}")
    # kr73_infer_clip_id_frame_index_pairs_under_these_models(
    #     final_model_id=model_id,
    #     set_of_clip_id_frame_index_pairs=set_of_clip_id_frame_index_pairs
    # )

sys.exit(0)
clip_id_and_frame_index_to_set_of_model_ids = defaultdict(set)
clip_id_and_frame_index_and_model_id_to_record = dict()

# for each model, infer all the 
for clip_id, frame_index in set_of_clip_id_frame_index_pairs:
    for model_id in models_in_order:
        folder = Path(f"~/a/preannotations/fixups/{clip_id}/{model_id}/").expanduser()
        assert folder.exists(), f"Folder {folder} does not exist"
        for original_file_path in folder.glob("*_original.jpg"):
            clip_id, frame_index = get_clip_id_and_frame_index_from_original_file_name(
                file_name=original_file_path.name
            )
            mask_file_path = (
                original_file_path.parent / f"{clip_id}_{frame_index:06d}_nonfloor.png"
            )
            assert mask_file_path.exists(), f"Mask file {mask_file_path} does not exist despite original file {original_file_path} existing"
            record = dict(
                clip_id=clip_id,
                frame_index=frame_index,
                model_id=model_id,
                original_file_path=original_file_path,
                mask_file_path=mask_file_path,
            )
            rows.append(
                record
            )
            ci = (clip_id, frame_index)
            clip_id_and_frame_index_to_set_of_model_ids[ci].add(model_id)
            set_of_clip_id_frame_index_pairs.add(
                ci
            )
            clip_id_and_frame_index_and_model_id_to_record[
                (clip_id, frame_index, model_id)
            ] = record
sorted_set_of_clip_id_frame_index_pairs = sorted(
    set_of_clip_id_frame_index_pairs,
    key=lambda x: (x[0], x[1])
)

# Print the sorted set of clip_id and frame_index pairs
print("Sorted set of clip_id and frame_index pairs:")

clip_id_and_frame_index_to_list_of_model_ids = dict()
for clip_id, frame_index in sorted_set_of_clip_id_frame_index_pairs:           
    clip_id_and_frame_index_to_list_of_model_ids[(clip_id, frame_index)] = sorted(
        list(clip_id_and_frame_index_to_set_of_model_ids[(clip_id, frame_index)]),
        key=lambda x: x
    )
# if you are navigating a spare matrix of elements with left and right up and down,
# there are a few ways to do it.
list_of_lists = []
for clip_id, frame_index in sorted_set_of_clip_id_frame_index_pairs:
    lst = []
    models_inferred_for_this_frame = clip_id_and_frame_index_to_list_of_model_ids[
        (clip_id, frame_index)
    ]
    models_inferred_for_this_frame = hd71_sort_as_this_list_is_sorted(
        list_defining_the_sort_order=models_in_order,
        list_to_sort=models_inferred_for_this_frame
    )
    for model_id in models_inferred_for_this_frame:
        record = clip_id_and_frame_index_and_model_id_to_record[
            (clip_id, frame_index, model_id)
        ]
        print(
            f"Clip ID: {record['clip_id']}, Frame Index: {record['frame_index']:06d}, Model ID: {record['model_id']}, "
            f"Original File Path: {record['original_file_path']}, Mask File Path: {record['mask_file_path']}"
        )
        mask_file_path = record['mask_file_path']
        original_file_path = record['original_file_path']
        javascript_mask_file_path = mask_file_path.relative_to(
            Path("~/a/preannotations/").expanduser()
        )
        javascript_original_file_path = original_file_path.relative_to(
            Path("~/a/preannotations/").expanduser()
        )
        javascript_mask_file_path_str = str(javascript_mask_file_path)
        javascript_original_file_path_str = str(javascript_original_file_path)
        name = f'{record["clip_id"]}_{record["frame_index"]:06d}_{record["model_id"]}'
        javascript_record_str = f'{{original: "{javascript_original_file_path_str}", mask: "{javascript_mask_file_path_str}", name: "{name}"}}'
       
        lst.append(
            javascript_record_str
        )
    list_of_lists.append(
        lst
    )

prepath = Path("pre.html").resolve()
pre_content = prepath.read_text()
postpath = Path("post.html").resolve()
post_content = postpath.read_text()

with open("index.html", "w") as f:
    f.write(pre_content)
    print("    const images = [", file=f)
    for lst in list_of_lists:
        print("        [", file=f)
        for x in lst:
            print(f"            {x},", file=f)
        print("        ],", file=f)
    print("    ];", file=f)
    f.write(post_content)


print_green("cd /shared/www")
print_green("python -m http.server 42857 --bind 127.0.0.1")

print_green("http://72.177.16.107:42857/index.html")