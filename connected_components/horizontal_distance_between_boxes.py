def horizontal_distance_between_boxes(a, b):
    a_x0, a_x1, a_y0, a_y1 = a
    b_x0, b_x1, b_y0, b_y1 = b
    if a_x0 < b_x0 and b_x0 < a_x1:
        return 0
    if a_x0 < b_x1 and b_x1 < a_x1:
        return 0
    distance = min(
        abs(a_x0 - b_x1),
        abs(a_x1 - b_x0),
    )
    return distance
