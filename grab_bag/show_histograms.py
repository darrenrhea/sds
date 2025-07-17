import pprint as pp
from prii import (
     prii
)
from ColorCorrector import (
     ColorCorrector
)
from pathlib import Path
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from matplotlib import pyplot as plt

def show_histograms(
    feature_name: str,
    kind: str,
    rgb_hwc_np_u8: np.ndarray
):
    assert kind in ["good", "bad"], f"{kind=} is not in ['good', 'bad']"
    assert feature_name in ["L", "A", "B"], f"{feature_name=} is not in ['L', 'A', 'B']"
    rgb_values = rgb_hwc_np_u8.reshape(-1, 3)

    lab_values = rgb2lab(rgb_values / 255.0)
    L = lab_values[:, 0]
    A = lab_values[:, 1]
    B = lab_values[:, 2]

    feature_name_to_feature = dict(
        L=L,
        A=A,
        B=B
    )

    feature_name_to_min =dict(
        L=0,
        A=-127,
        B=-127
    )
    feature_name_to_max =dict(
        L=100,
        A=128,
        B=128
    )
    theoretical_min = feature_name_to_min[feature_name]
    theoretical_max = feature_name_to_max[feature_name]
    feature = feature_name_to_feature[feature_name]
    feature_min = feature_name_to_min[feature_name]
    feature_max = feature_name_to_max[feature_name]
    print(f"{feature_name} on {kind} has these statistics:")
    the_min = np.min(feature)
    print(f"q0=min={the_min}")
    q25 = np.quantile(a=feature, q=0.25)
    q75 = np.quantile(a=feature, q=0.75)
    print(f"{q25=}")
    the_median = np.median(feature)
    print(f"median = {the_median}")
    print(f"{q75=}")
    the_max = np.max(feature)
    print(f"q100=max={the_max}")
   
    
    q = np.zeros(shape=(101,))
    for k in range(0, 100+1):
        if k==0:
            q[0]  = theoretical_min
        elif k==100:
            q[100] = theoretical_max
        else:
            q[k] = np.quantile(a=feature, q=k/100)

    # for k in range(0, 100+1, 10):
    #     print(f"{k=}, {q[k]=}")

    plt.figure(figsize=(30, 7))
    plt.hist(feature, 256, [feature_min, feature_max], color = 'r')
    plt.ylim([0, 160000])
    plt.title(f"The {kind} image's histogram of {feature_name}")
    file_name = f"histograms/{feature_name}_{kind}_histogram.png"
    plt.savefig(file_name)
    prii(file_name)

    answer = dict(
        min=the_min,
        q25=q25,
        median=the_median,
        q75=q75,
        max=the_max,
    )
    return q, answer

def main():
    feature_name = "L"
    good_bad_image_pairs = [
        [
            "/media/drhea/muchspace/clips/munich2023-10-05-1080i-yadif/frames/munich2023-10-05-1080i-yadif_100089_original.jpg",
            "/media/drhea/muchspace/clips/youtubeAmiCuoupzPQ/frames/youtubeAmiCuoupzPQ_001726_original.jpg",
        ],
        [
            "/media/drhea/muchspace/clips/munich2024-01-09-1080i-yadif/frames/munich2024-01-09-1080i-yadif_010450_original.jpg",
            "/media/drhea/muchspace/clips/youtubeAmiCuoupzPQ/frames/youtubeAmiCuoupzPQ_001600_original.jpg"
        ]
    ]
    
    which_pair = 0

    kind_to_image_path = {
        "good": Path(good_bad_image_pairs[which_pair][0]),
        "bad": Path(good_bad_image_pairs[which_pair][1]),
    }

    kind_to_rgb_hwc_np_u8 = {}
    kind_to_trunc_rgb_hwc_np_u8 = {}
    for kind in ["good", "bad"]:
        kind_to_rgb_hwc_np_u8[kind] = open_as_rgb_hwc_np_u8(
            image_path=kind_to_image_path[kind]
        )
        image_path = kind_to_image_path[kind]
        print(f"The {kind} image is: {image_path}:")
        
        rgb_hwc_np_u8 = kind_to_rgb_hwc_np_u8[kind]

        prii(rgb_hwc_np_u8)
        i_max = 500
        kind_to_trunc_rgb_hwc_np_u8[kind] = rgb_hwc_np_u8[:i_max, :, :]
        print("Note: we only use this piece of it for the histogram:")
        prii(kind_to_trunc_rgb_hwc_np_u8[kind])

    feature_names = ["L", "A", "B"]
    feature_name_and_kind_to_quantiles = {}
    for feature_name in feature_names:
        for kind in ["good", "bad"]:   
            print(f"\n\n\n\nFor the {kind}-looking image, feature {feature_name} has these statistics:")
            q, answer = show_histograms(
                kind=kind,
                feature_name=feature_name,
                rgb_hwc_np_u8=kind_to_trunc_rgb_hwc_np_u8[kind]
            )
            feature_name_and_kind_to_quantiles[
                (feature_name, kind)
            ] = q
            print(f"For the {kind}-looking image, feature {feature_name} has these quantiles:")
            for k in range(0, 100+1, 10):
                print(f"{k=}, {q[k]=}")

    for feature_name in feature_names:
        xs = feature_name_and_kind_to_quantiles[
            (feature_name, "bad")
        ]
        ys = feature_name_and_kind_to_quantiles[
            (feature_name, "good")
        ]
        plt.figure(figsize=(10, 10))
        plt.plot(xs, ys, 'b-')
        plt.title(f"A potential map from bad to good value for {feature_name}")
        file_name = f"histograms/{feature_name}_potential_bad_to_good_map.png"
        plt.savefig(file_name)
        prii(file_name)


if __name__ == "__main__":
    main()