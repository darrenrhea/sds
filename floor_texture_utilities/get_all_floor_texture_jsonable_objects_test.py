import pprint


from get_all_floor_texture_jsonable_objects import (
     get_all_floor_texture_jsonable_objects
)


def test_get_all_floor_texture_jsonable_objects_1():
    ans = get_all_floor_texture_jsonable_objects(
        slow_check=True
    )
    pprint.pprint(ans)
    

if __name__ == "__main__":
    test_get_all_floor_texture_jsonable_objects_1()