from hd71_sort_as_this_list_is_sorted import hd71_sort_as_this_list_is_sorted


def test_hd71_sort_as_this_list_is_sorted_1():
    """
    Test the hd71_sort_as_this_list_is_sorted function.
    """
    list_defining_the_sort_order = ['a', 'b', 'c', 'd']
    list_to_sort = ['c', 'a', 'b', 'd']
    
    sorted_list = hd71_sort_as_this_list_is_sorted(
        list_defining_the_sort_order,
        list_to_sort
    )
    
    assert sorted_list == ['a', 'b', 'c', 'd'], f"Expected ['a', 'b', 'c', 'd'], got {sorted_list}"