import PIL
import PIL.Image
from print_image_in_iterm2 import print_image_in_iterm2
import numpy as np


def test_print_image_in_iterm2_1():
    image_pil = PIL.Image.open("fixtures/image.jpg")
    print("providing title works:")

    print_image_in_iterm2(
        image_pil=image_pil,
        title="monkey.png"
    )

def test_print_image_in_iterm2_2():
    image_pil = PIL.Image.open("fixtures/dice.png")

    print_image_in_iterm2(
        image_pil=image_pil,
        title="dice.png"
    )


def test_print_image_in_iterm2_3():
    image_pil = PIL.Image.open("fixtures/image.jpg")

    print_image_in_iterm2(
        image_pil=image_pil
    )

def test_print_image_in_iterm2_4():
    image_pil = PIL.Image.open("fixtures/image.jpg")

    print_image_in_iterm2(
        image_pil=image_pil
    )

def test_print_image_in_iterm2_5():
    image_pil = PIL.Image.open("fixtures/image.jpg")
    rgb_np_u8 = np.array(image_pil)

    print_image_in_iterm2(
        rgb_np_uint8=rgb_np_u8
    )

def test_print_image_in_iterm2_6():
    image_pil = PIL.Image.open("fixtures/image.jpg")
    rgb_np_u8 = np.array(image_pil)

    print_image_in_iterm2(
        rgb_np_uint8=rgb_np_u8,
        scale=2
    )

def test_print_image_in_iterm2_7():
    print("bgra_np_u8 works:")
    image_pil = PIL.Image.open("fixtures/dice.png").convert("RGBA")
    bgra_np_u8 = np.array(image_pil)[:, :, ::-1]

    print_image_in_iterm2(
        bgra_np_u8=bgra_np_u8,
        scale=1
    )

def test_print_image_in_iterm2_8():
    print("bgra_np_u8 works:")
    image_path = "fixtures/dice.png"
    image_pil = PIL.Image.open(image_path).convert("RGBA")
    bgra_np_u8 = np.array(image_pil)[:, :, [2, 1, 0, 3]]

    print_image_in_iterm2(
        bgra_np_u8=bgra_np_u8,
        scale=1
    )


def main():
    test_print_image_in_iterm2_1()
    test_print_image_in_iterm2_2()
    test_print_image_in_iterm2_3()
    test_print_image_in_iterm2_4()
    test_print_image_in_iterm2_5()
    test_print_image_in_iterm2_6()
    test_print_image_in_iterm2_7()
    test_print_image_in_iterm2_8()


if __name__ == '__main__':
    main()
