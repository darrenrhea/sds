from cover_subrectangles import cover_subrectangles
from cover_subrectangles import cover_subrectangle
import numpy as np
import pprint as pp

def check_this_is_a_fair_question_for_cover_subrectangle(
    photo_width,
    photo_height,
    nn_input_width,
    nn_input_height,
    subrectangle
):
    pixels = np.zeros(shape=(photo_height, photo_width), dtype=bool)
    # make every pixel that needs to be covered have value 1:
    assert nn_input_height > 0 
    assert nn_input_width > 0
    assert nn_input_height <= photo_height
    assert nn_input_width <= photo_width
    i_min, i_max, j_min, j_max = subrectangle["i_min"], subrectangle["i_max"], subrectangle["j_min"], subrectangle["j_max"]
    assert i_min >= 0
    assert j_min >= 0
    assert i_min < i_max
    assert j_min < j_max
    assert i_max <= photo_height
    assert j_max <= photo_width

def check_this_is_a_fair_question_for_cover_subrectangles(
    photo_width,
    photo_height,
    nn_input_width,
    nn_input_height,
    list_of_subrectangles
):
    pixels = np.zeros(shape=(photo_height, photo_width), dtype=bool)
    # make every pixel that needs to be covered have value 1:
    assert nn_input_height > 0 
    assert nn_input_width > 0
    assert nn_input_height <= photo_height
    assert nn_input_width <= photo_width
    for rect in list_of_subrectangles:
        i_min, i_max, j_min, j_max = rect["i_min"], rect["i_max"], rect["j_min"], rect["j_max"]
        assert i_min >= 0
        assert j_min >= 0
        assert i_min < i_max
        assert j_min < j_max
        assert i_max <= photo_height
        assert j_max <= photo_width
 
def randomly_generate_a_question_for_cover_subrectangle():
    photo_width = np.random.randint(1, 2000)
    photo_height = np.random.randint(1, 2000)
    nn_input_width = np.random.randint(1, photo_width + 1)
    nn_input_height = np.random.randint(1, photo_height + 1)

    i_min = np.random.randint(0, photo_height)
    j_min = np.random.randint(0, photo_width)
    i_max = np.random.randint(i_min + 1, photo_height + 1)
    j_max = np.random.randint(j_min + 1, photo_width + 1)
    subrectangle = dict(i_min=i_min, i_max=i_max, j_min=j_min, j_max=j_max)

    question = dict(
        photo_width=photo_width,
        photo_height=photo_height,
        nn_input_width=nn_input_width,
        nn_input_height=nn_input_height,
        subrectangle=subrectangle
    )

    check_this_is_a_fair_question_for_cover_subrectangle(**question)
    
    return question

def randomly_generate_a_question_for_cover_subrectangles():
    photo_width = np.random.randint(1, 2000)
    photo_height = np.random.randint(1, 2000)
    nn_input_width = np.random.randint(1, photo_width + 1)
    nn_input_height = np.random.randint(1, photo_height + 1)

    list_of_subrectangles = []
    num_subrectangles = np.random.randint(0, 4)
    for _ in range(num_subrectangles):
        i_min = np.random.randint(0, photo_height)
        j_min = np.random.randint(0, photo_width)
        i_max = np.random.randint(i_min + 1, photo_height + 1)
        j_max = np.random.randint(j_min + 1, photo_width + 1)
        rect = dict(i_min=i_min, i_max=i_max, j_min=j_min, j_max=j_max)
        list_of_subrectangles.append(rect)

    question = dict(
        photo_width=photo_width,
        photo_height=photo_height,
        nn_input_width=nn_input_width,
        nn_input_height=nn_input_height,
        list_of_subrectangles=list_of_subrectangles
    )

    check_this_is_a_fair_question_for_cover_subrectangles(**question)
    
    return question

def check_if_cover_subrectangle_works_on_this_input(
    photo_width,
    photo_height,
    nn_input_width,
    nn_input_height,
    subrectangle
):
    pixels = np.zeros(shape=(photo_height, photo_width), dtype=bool)
    # make every pixel that needs to be covered have value 1:
    number_of_tiles_needed_to_cover = 0
    
    i_min, i_max, j_min, j_max = subrectangle["i_min"], subrectangle["i_max"], subrectangle["j_min"], subrectangle["j_max"]
    pixels[i_min:i_max, j_min:j_max] = 1
    subrectangle_height = i_max  - i_min
    subrectangle_width = j_max  - j_min
    min_num_wide = (subrectangle_width + nn_input_width - 1) // nn_input_width
    min_num_tall = (subrectangle_height + nn_input_height - 1) // nn_input_height
    number_of_tiles_needed_to_cover = min_num_wide * min_num_tall
    
    cover = cover_subrectangle(
        photo_width=photo_width,
        photo_height=photo_height,
        nn_input_width=nn_input_width,
        nn_input_height=nn_input_height,
        subrectangle=subrectangle
    )

    assert len(cover) <= number_of_tiles_needed_to_cover
    row_points = []
    column_points = []
    for subrectangle in cover:
        print(f"subrectangle {subrectangle}")
        row_min = subrectangle["i_min"]
        row_max = subrectangle["i_max"]
        column_min = subrectangle["j_min"]
        column_max = subrectangle["j_max"]
        for i in range(row_min, row_max + 1):
            row_points.append(i)
        for j in range(column_min, column_max + 1):
            column_points.append(j)
    
    # print(f"row points {row_points}")
    # print(f"column points {column_points}")
    assert set(row_points).issuperset(set(range(subrectangle["i_min"], subrectangle["i_max"] + 1)))
    assert set(column_points).issuperset(set(range(subrectangle["j_min"], subrectangle["j_max"] + 1))) 

def check_if_cover_subrectangles_works_on_this_input(
    photo_width,
    photo_height,
    nn_input_width,
    nn_input_height,
    list_of_subrectangles
):
    pixels = np.zeros(shape=(photo_height, photo_width), dtype=bool)
    # make every pixel that needs to be covered have value 1:
    number_of_tiles_needed_to_cover = 0
    for rect in list_of_subrectangles:
        i_min, i_max, j_min, j_max = rect["i_min"], rect["i_max"], rect["j_min"], rect["j_max"]
        pixels[i_min:i_max, j_min:j_max] = 1
        rect_height = i_max  - i_min
        rect_width = j_max  - j_min
        min_num_wide = (rect_width + nn_input_width - 1) // nn_input_width
        min_num_tall = (rect_height + nn_input_height - 1) // nn_input_height
        number_of_tiles_needed_to_cover += min_num_wide * min_num_tall
    
    cover = cover_subrectangles(
        photo_width=photo_width,
        photo_height=photo_height,
        nn_input_width=nn_input_width,
        nn_input_height=nn_input_height,
        list_of_subrectangles=list_of_subrectangles
    )

    assert len(cover) <= number_of_tiles_needed_to_cover

    # use the cover to set whatever is covered back zero.
    for rect in cover:
        i_min, i_max, j_min, j_max = rect["i_min"], rect["i_max"], rect["j_min"], rect["j_max"]
        pixels[i_min:i_max, j_min:j_max] = 0

    # if it really 
    assert np.all(pixels == 0)

    # Check that the cover consists entirely of nn_input_height x nn_input_width things (currently 224 x 224):
    for rect in cover:
        i_min, i_max, j_min, j_max = rect["i_min"], rect["i_max"], rect["j_min"], rect["j_max"]
        assert (
            i_max - i_min == nn_input_height
        ), f"one of the cover's rectangles is not the right height, each must be {nn_input_height} high."
        assert (
            j_max - j_min == nn_input_width
        ), f"one of the cover's rectangles is not the right width, each must be {nn_input_width} width."
        assert i_min >= 0
        assert j_min >= 0
        assert i_min < i_max
        assert j_min < j_max
        assert i_max <= photo_height
        assert j_max <= photo_width

    print("cover_subrectangles passed")

def test_cover_subrectangle():
    question = randomly_generate_a_question_for_cover_subrectangle()
    pp.pprint(question)
    check_if_cover_subrectangle_works_on_this_input(**question)

def test_cover_subrectangles():
    question = dict(
        photo_width=1920,
        photo_height=1080,
        nn_input_width=224,
        nn_input_height=224,
        list_of_subrectangles = [
            dict(i_min=0, i_max=224, j_min=0, j_max=224),
        ]
    )

    question = randomly_generate_a_question_for_cover_subrectangles()
    pp.pprint(question)
    check_if_cover_subrectangles_works_on_this_input(**question)


if __name__ == "__main__":
    test_cover_subrectangle()
