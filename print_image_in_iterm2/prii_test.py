import PIL
import PIL.Image
from prii import prii
import numpy as np
from pathlib import Path


def test_prii_1():
    image_pil = PIL.Image.open("fixtures/image.jpg")
    print("PIL.Images work:")

    prii(image_pil)

def test_prii_2():
    image_pil = PIL.Image.open("fixtures/dice.png")
    print("alpha PIL.Images work:")
    prii(image_pil)


def test_prii_3():
    s = "fixtures/image.jpg"
    print("file path strings work:")
    prii(s)

def test_prii_4():
    p = Path("fixtures/image.jpg")
    print("Path objects work:")
    prii(p)

def test_prii_5():
    image_pil = PIL.Image.open("fixtures/image.jpg")
    rgb_np_u8 = np.array(image_pil)
    print("rgb numpy arrays work without hint")
    prii(rgb_np_u8)

def test_prii_6():
    image_pil = PIL.Image.open("fixtures/dice.png")
    rgba_np_u8 = np.array(image_pil)
    print("rgba numpy arrays work without hint")
    prii(rgba_np_u8)



def test_prii_7():
    print("If frontmost die is red, prii on bgra_np_u8 works with hint \"bgr\"")
    image_path = "fixtures/dice.png"
    image_pil = PIL.Image.open(image_path).convert("RGBA")
    bgra_np_u8 = np.array(image_pil)[:, :, [2, 1, 0, 3]]

    prii(bgra_np_u8, hint="bgr", scale=2)

def test_prii_8():
    image_pil = PIL.Image.open("fixtures/image.jpg")
    print("PIL.Images work with scaling:")

    prii(image_pil, scale=3.0)


def main():
    test_prii_1()
    test_prii_2()
    test_prii_3()
    test_prii_4()
    test_prii_5()
    test_prii_6()
    test_prii_7()
    test_prii_8()


if __name__ == '__main__':
    main()
