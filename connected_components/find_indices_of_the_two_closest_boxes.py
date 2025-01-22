from horizontal_distance_between_boxes import (
     horizontal_distance_between_boxes
)

def find_indices_of_the_two_closest_boxes(
    bounding_boxes
):
    closest_so_far = float("inf")
    i_closest_so_far = None
    j_closest_so_far = None
    for i in range(len(bounding_boxes)):
        for j in range(i + 1, len(bounding_boxes)):
            d = horizontal_distance_between_boxes(bounding_boxes[i], bounding_boxes[j])
            if d < closest_so_far:
                closest_so_far = d
                i_closest_so_far = i
                j_closest_so_far = j
    
    return i_closest_so_far, j_closest_so_far

