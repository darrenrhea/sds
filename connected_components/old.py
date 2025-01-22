"""
We cut this into pieces.
"""
import sys
import cv2
import numpy as np
from warp_image import warp_image
import PIL.Image
from pathlib import Path
from print_image_in_iterm2 import print_image_in_iterm2
import pprint as pp
import dataclasses



class RectangleSummer(object):
    def __init__(self, binary_mask_np_uint8):
        """
        Suppose you wanted to know the number of Trues in a rectangular region of a binary mask,
        for many different rectangular regions.  You need something like this.
        """
        height = binary_mask_np_uint8.shape[0]
        width = binary_mask_np_uint8.shape[1]
        self.y = np.zeros(shape=(height+1, width+1), dtype=np.int32)
        self.y[1:, 1:] = np.cumsum(np.cumsum(binary_mask_np_uint8, axis=0), axis=1)

    def sum(self, x_min, xMax, y_min, yMax):
        """
        returns the sum over x_min <= x < xMax and y_min <= y < yMax of the binary mask
        """
        assert x_min >= 0
        assert y_min >= 0
        assert xMax > x_min
        assert yMax > y_min
        assert xMax <= self.y.shape[1]
        assert yMax <= self.y.shape[0]
        return self.y[yMax, xMax] - self.y[yMax, x_min] - self.y[y_min, xMax] + self.y[y_min, x_min]

    def frac(self, x_min, xMax, y_min, yMax):
        return self.sum(x_min, xMax, y_min, yMax) / ((xMax - x_min) * (yMax - y_min))


# 0 1 1 1 1
# 0 0 1 1 1
# 0 0 0 1 1
# 1 1 0 0 1
# 1 1 1 0 0
def test_RectangleSummer():
    b = np.array(
        [
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [1, 1, 0, 0, 1],
            [1, 1, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    summer = RectangleSummer(b)
    assert summer.sum(0, 2, 0, 2) == 2
    assert summer.sum(0, 2, 0, 5) == 6
    assert summer.sum(0, 5, 2, 4) == 5
    assert summer.sum(0, 5, 0, 5) == 16

test_RectangleSummer()

def make_smaller(x0x1y0y1, manner):
    assert manner in [0, 1, 2, 3]
    x0, x1, y0, y1 = x0x1y0y1
    failed = True
    new_x0x1y0y1 = None
    if manner == 0:
        if x0 < x1 - 1:
            new_x0x1y0y1 = (x0 + 1, x1, y0, y1)
            failed = False
        return new_x0x1y0y1, failed
    if manner == 1:
        if x0 < x1 - 1:
            new_x0x1y0y1 = (x0, x1 - 1, y0, y1)
            failed = False
        return new_x0x1y0y1, failed
    if manner == 2:
        if y0 < y1 - 1:
            new_x0x1y0y1 = (x0, x1, y0 + 1, y1)
            failed = False
        return new_x0x1y0y1, failed
    if manner == 3:
        if y0 < y1 - 1:
            new_x0x1y0y1 = (x0, x1, y0, y1 - 1)
            failed = False
        return new_x0x1y0y1, failed


def shrink(x0x1y0y1, summer):
    current_sum = summer.sum(*x0x1y0y1)
    print(f"{current_sum=}")
    while True:
        at_least_one_manner_succeeded = False
        # try 4 ways to make the rectangle smaller:
        for manner in range(4):
            new_x0x1y0y1, failed = make_smaller(
                x0x1y0y1=x0x1y0y1,
                manner=manner
            )
            if failed:
                continue

            new_sum = summer.sum(*new_x0x1y0y1)
            print(f"{new_x0x1y0y1=}")
            print(f"{new_sum=}")
            if new_sum >= current_sum:
                x0x1y0y1 = new_x0x1y0y1
                at_least_one_manner_succeeded = True
                current_sum = new_sum
                break
            else:
               print("Not better")
        if not at_least_one_manner_succeeded:
            break
    return x0x1y0y1


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

def union_two_boxes_together(a, b):
    a_x0, a_x1, a_y0, a_y1 = a
    b_x0, b_x1, b_y0, b_y1 = b
    x0 = min(a_x0, b_x0)
    x1 = max(a_x1, b_x1)
    y0 = min(a_y0, b_y0)
    y1 = max(a_y1, b_y1)
    return x0, x1, y0, y1
   
def find_indices_of_the_two_closest_boxes(bounding_boxes):
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


def union_until_only_3_boxes_remain(bounding_boxes):
    current_bounding_boxes = bounding_boxes.copy()
    while len(current_bounding_boxes) > 3:
        # print("There are too many boxes:")
        # print(current_bounding_boxes)
        i, j = find_indices_of_the_two_closest_boxes(current_bounding_boxes)
        # print(f"Unioning the two closest boxes, namely {current_bounding_boxes[i]} and {current_bounding_boxes[j]} together")
        union_box = union_two_boxes_together(current_bounding_boxes[i], current_bounding_boxes[j])
        # print(f"The union is {union_box}")
        new_bounding_boxes = [b for k, b in enumerate(current_bounding_boxes) if k != i and k != j]
        new_bounding_boxes.append(union_box)
        current_bounding_boxes = new_bounding_boxes
    return current_bounding_boxes


def find_rectangles_from_segmentation(
    original_path,  # the original image to insert ads into
    mask_path,  # where to find the mask that cuts out the LED screens
    out_path  # where to save the result
):

    nonled_mask_np_bgra_uint8 = cv2.imread(
        filename=str(mask_path),
        flags=cv2.IMREAD_UNCHANGED  # without this, it will not load the alpha channel
    )
   
    original_np = np.array(PIL.Image.open(original_path).convert("RGB"))

    height = nonled_mask_np_bgra_uint8.shape[0]
    width = nonled_mask_np_bgra_uint8.shape[1]

    assert (
        nonled_mask_np_bgra_uint8.shape[2] == 4
    ), f"ERROR: {mask_path} is supposed to be an RGBA PNG, which would have 4 channels"

    alpha_uint8 = nonled_mask_np_bgra_uint8[:, :, 3]
    # print_image_in_iterm2(grayscale_np_uint8=alpha_uint8)
    # flip it to indicate where the LED screens are:
    binary = (alpha_uint8 < 128).astype(np.uint8)
    summer = RectangleSummer(binary)

    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    
    bounding_boxes_with_measure = []
    for i in range(numLabels):
        # print(f"{i=}")
        # print_image_in_iterm2(grayscale_np_uint8=(labels == i).astype(np.uint8) * 255)
        x0, y0, width, height, measure = stats[i]
        x1 = x0 + width
        y1 = y0 + height
        # print(f"{x0=}, {x1=}, {y0=}, {y1=}, {measure=}")
        if i > 0 and measure > 500:
            bounding_boxes_with_measure.append((x0, x1, y0, y1, measure))
    bounding_boxes_with_measure = sorted(bounding_boxes_with_measure, key=lambda x: x[4], reverse=True)
    bounding_boxes = [x[:4] for x in bounding_boxes_with_measure]

    bounding_boxes = union_until_only_3_boxes_remain(bounding_boxes)

    pp.pprint(bounding_boxes)
    for k in range(len(bounding_boxes)):
        x0, x1, y0, y1 = bounding_boxes[k]
        original_np[y0:y1, x0:x1, 0:2] = 0
    print_image_in_iterm2(rgb_np_uint8=original_np)
   

    # x0 = np.random.randint(0, 1280 - 1)
    # x1 = np.random.randint(x0 + 1, 1280)
    # y0 = np.random.randint(0, 720 - 1)
    # y1 = np.random.randint(y0 + 1, 720)

    # x0, x1 = 259, 990
    # y0, y1 = 191, 423

    # x0x1y0y1 = (x0, x1, y0, y1)
    # print(f"Starting with [{x0}, {x1}] . [{y0}, {y1}]")
    # x0x1y0y1 = shrink(x0x1y0y1, summer=summer)
    # x0, x1, y0, y1 = x0x1y0y1
    # print(f"We shrink to [{x0}, {x1}] . [{y0}, {y1}]")



    

    bgr_uint8 = np.array(PIL.Image.open(str(original_path)))
    foreground_bgr_float = bgr_uint8.astype(float)





if __name__ == "__main__":
    dir = Path("~/busch_rec_screens_temp").expanduser()
    for path in dir.glob("*.jpg"):
        print(f"{path=}")
        image_name = path.stem
        print(f"{image_name=}")

        original_path = dir / f"{image_name}.jpg"
        mask_path = dir / f"{image_name}_nonfloor.png"
            
        find_rectangles_from_segmentation(
            original_path=original_path,
            mask_path=mask_path,
            out_path="out.png"
        )