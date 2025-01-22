from get_world_coordinate_descriptors_of_ad_placements import (
     get_world_coordinate_descriptors_of_ad_placements
)

from AdPlacementDescriptor import (
     AdPlacementDescriptor
)


def test_get_world_coordinate_descriptors_of_ad_placements_1():
    clip_id = "slgame1"
    with_floor_as_giant_ad = False
    overcover_by = 0.0

    ad_placement_descriptors = get_world_coordinate_descriptors_of_ad_placements(
        clip_id=clip_id,
        with_floor_as_giant_ad=with_floor_as_giant_ad,
        overcover_by=overcover_by,
    )
    for ad_placement_descriptor in ad_placement_descriptors:
        assert isinstance(ad_placement_descriptor, AdPlacementDescriptor)
        print(ad_placement_descriptor)


if __name__ == "__main__":
    test_get_world_coordinate_descriptors_of_ad_placements_1()
    print("get_world_coordinate_descriptors_of_ad_placements_test.py: All tests pass")