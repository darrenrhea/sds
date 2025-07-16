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
