from prii import (
     prii
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from pathlib import Path
from quantize_colors_via_kmeans import (
     quantize_colors_via_kmeans
)

def test_quantize_colors_via_means_1():
    image_path = Path(
        #"/Users/darrenrhea/r/munich_led_videos/TURKISH AIRLINES/FINAL_THY_New LED boards BAYERN_1920x1080/1152x144/00000.png"
        "/Users/darrenrhea/Downloads/image.png"
    )
    prii(image_path)

    rgb = open_as_rgb_hwc_np_u8(image_path)
    rgb_values = rgb.reshape((-1, 3))
    indices, centroids_u8 = quantize_colors_via_kmeans(
        rgb_values,
        num_colors_to_quantize_to=14,
    )
    indices_2d = indices.reshape(rgb.shape[:2])
    name_to_index = dict()
    name_to_xy = dict(
        inner_disk=[984, 309],
        ring=[1195, 742],
        outer_maroon=[1473, 188],
        gray_apron=[65,35],
        goldest=[813, 464],
        less_gold=[130, 450],
        streak=[152, 151],
        withinringstreak=[334, 300],
        black_trophy=[880, 338],

    )
    
    for name, xy in name_to_xy.items():
        name_to_index[name] = indices_2d[xy[1], xy[0]]
    
    #name_to_index["ring"] = indices_2d[309, 984]
    name_to_color = dict(
        inner_disk=[244, 37, 49],
        ring=[192, 34, 50],
        outer_maroon=[161, 35, 50],
        gray_apron=[56, 56, 58],
        goldest=[255, 231, 145],
        less_gold=[228, 183, 114],
        streak=[220, 95, 100],
        withinringstreak=[255, 95, 100],
        black_trophy=[38, 25, 20],
    )

    assert indices.shape[0] == rgb_values.shape[0]
    print(centroids_u8)
    for name, index in name_to_index.items():
        centroids_u8[index, :] = name_to_color[name]
    
    result = centroids_u8[indices].reshape(rgb.shape)
    print(centroids_u8)
    print(result.shape)
    print(result.dtype)
    prii(result, out=Path("temp.png"))
    

if __name__ == "__main__":
    test_quantize_colors_via_means_1()