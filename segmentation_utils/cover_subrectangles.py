from get_subrectangles_that_need_masking import get_subrectangles_that_need_masking


def cover_subrectangle(
    photo_width,
    photo_height,
    nn_input_width,
    nn_input_height,
    subrectangle
):
    subrectangle_cover = []
    i_min = subrectangle["i_min"]
    i_max = subrectangle["i_max"]
    j_min = subrectangle["j_min"]
    j_max = subrectangle["j_max"]
    columns = [x for x in range(j_min, max(j_min + nn_input_width, j_max - nn_input_width), nn_input_width)]
    if j_max - nn_input_width > j_min:
        columns += [j_max - nn_input_width]
    rows = [y for y in range(i_min, max(i_min + nn_input_height, i_max - nn_input_height), nn_input_height)]
    if i_max - nn_input_height > i_min:
        rows += [i_max - nn_input_height]
    all_chunks = [(column, row) for column in columns for row in rows]
    for cntr, (column_min, row_min) in enumerate(all_chunks):
        column_max = column_min + nn_input_width
        if column_max > photo_width:
            over_max = column_max - photo_width
            column_min = column_min - over_max
            column_max = photo_width
        row_max = row_min + nn_input_height
        if row_max > photo_height:
            over_max = row_max - photo_height
            row_min = row_min - over_max
            row_max = photo_height

        subrectangle_cover.append(
            dict(
                i_min=row_min,
                i_max=row_max,
                j_min=column_min,
                j_max=column_max
            )
        )
    return subrectangle_cover


def cover_subrectangles(
    photo_width,
    photo_height,
    nn_input_width,
    nn_input_height,
    list_of_subrectangles
):
    """
    List of subrectangles is a list of ad placements.
    """
    all_subrectangles = []
    for sub_rectangle in list_of_subrectangles:
        subrectangle_cover = cover_subrectangle(photo_width, photo_height, nn_input_width, nn_input_height, sub_rectangle)
        all_subrectangles.extend(subrectangle_cover)

    return all_subrectangles
