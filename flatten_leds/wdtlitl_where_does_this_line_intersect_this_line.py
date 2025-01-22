def wdtlitl_where_does_this_line_intersect_this_line(
    line1_as_a_list_of_two_points,
    line2_as_a_list_of_two_points
):
    """
    Where does this line intersect this line?

    Parameters
    ----------
    line1_as_a_list_of_two_points : list
        The first line, defined by two points.
    line2_as_a_list_of_two_points : list
        The second line, defined by two points.

    Returns
    -------
    list
        The point of intersection.
    """
    a, b = line1_as_a_list_of_two_points
    c, d = line2_as_a_list_of_two_points
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    x4, y4 = d
    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    return [x, y]