from get_valid_cutout_kinds import (
     get_valid_cutout_kinds
)
from PasteableCutout import (
     PasteableCutout
)
from typing import (
     Dict,
     List
)


def group_cutouts_by_kind(
    sport: str,
    cutouts: List[PasteableCutout]
) -> Dict[str, List[PasteableCutout]]:
    """
    We want to be able to adjust how many per kind of cutout.
    Also, the augmentation should be different per kind of cutout.
    """
    cutouts_by_kind = dict()
    valid_cutout_kinds = get_valid_cutout_kinds(sport=sport)
    for kind in valid_cutout_kinds:
        cutouts_by_kind[kind] = [
            x
            for x in cutouts
            if x.kind == kind
        ]
    
        print(f"Found {len(cutouts_by_kind[kind])} cutouts of kind {kind}.")
    return cutouts_by_kind