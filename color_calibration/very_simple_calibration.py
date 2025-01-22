import numpy as np

red_from_to_pairs = [
    [154, 146],  # google pixel
    [76, 117],  # ATT against ~/r/slgame1_led/.flat/slgame1_057500_original.jpg
    [34, 57],   # 2K25 against ~/r/slgame1_led/.flat/slgame1_019000_original.jpg
    [208, 232],
    [228, 245], # gatorade against ~/r/slgame1_led/.flat/slgame1_095500_original.jpg
]
green_from_to_pairs = [
    [31, 54],    # 2K25 against ~/r/slgame1_led/.flat/slgame1_019000_original.jpg
    [168, 252],  # ATT against ~/r/slgame1_led/.flat/slgame1_057500_original.jpg

    [135, 244],  # gatorade against ~/r/slgame1_led/.flat/slgame1_095500_original.jpg
]
blue_from_to_pairs = [
    [32, 65],    # 2K25 against ~/r/slgame1_led/.flat/slgame1_019000_original.jpg   
    [44, 90],  # kia
    [227, 252],  # ATT against ~/r/slgame1_led/.flat/slgame1_057500_original.jpg
    [59, 69], # gatorade against ~/r/slgame1_led/.flat/slgame1_095500_original.jpg

]

for channel_index, from_to_pairs in enumerate([
    red_from_to_pairs,
    green_from_to_pairs,
    blue_from_to_pairs,
]):

    from_to_pairs_u8 = np.array(
        from_to_pairs,
        dtype=np.uint8
    )

    from_to_pairs_nonlinear_f32 = from_to_pairs_u8.astype(np.float32) / 255.0


    from_to_pairs_linear_f32 = from_to_pairs_nonlinear_f32 ** 2.2

    x0 = from_to_pairs_linear_f32[0][0]
    y0 = from_to_pairs_linear_f32[0][1]
    x1 = from_to_pairs_linear_f32[1][0]
    y1 = from_to_pairs_linear_f32[1][1]
    slope = (y1 - y0) / (x1 - x0)
    boost = y0 - slope * x0
    c = "rgb"[channel_index]

    print(f"{c}m, {c}_boost ={slope}, {boost}")






