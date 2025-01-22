from get_all_the_approved_annotations_from_this_repo import (
     get_all_the_approved_annotations_from_this_repo
)
from pathlib import Path
from typing import List



def get_approved_annotations_from_these_repos(
    repo_ids_to_use: List[str],
    pull_again: bool = False
):
    """
    We start with approved annotations when making fake data.
    This provides a very usable list of approved annotations.
    """
    # clip_id_to_clip_info = get_clip_id_to_clip_info()
    

    approved_annotations = []
    for repo_id in repo_ids_to_use:
        repo_dir = Path(
            f"~/r/{repo_id}"
        ).expanduser()
        
        approved_annotations_from_this_repo = (
            get_all_the_approved_annotations_from_this_repo(
                repo_dir=repo_dir,
                pull_again=pull_again,
            )
        )
        print(f"{repo_id=} has {len(approved_annotations_from_this_repo)} approved")
        for approved_annotation in approved_annotations_from_this_repo:
            approved_annotation["repo_id"] = repo_id

        approved_annotations.extend(
            approved_annotations_from_this_repo
        )
    
    approved_annotations.sort(
        key=lambda x: (x["clip_id"], x["frame_index"])
    )
    return approved_annotations
