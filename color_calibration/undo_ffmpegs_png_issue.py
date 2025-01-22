from sRGB_to_CIE_linear_XYZ import (
     sRGB_to_CIE_linear_XYZ
)
from CIE_linear_XYZ_to_sRGB import (
     CIE_linear_XYZ_to_sRGB
)
from write_rgb_hwc_np_u8_to_png import (
     write_rgb_hwc_np_u8_to_png
)
from pathlib import Path
import numpy as np
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from sklearn import linear_model

# without linearization, we get:
# L_1_error=1.7671806587410954
# L_2_error=2.4973688096259465
# L_infinity_error=74.05349092994223


print(
    f"{CIE_linear_XYZ_to_sRGB(np.array([[0.4124564, 0.2126729, 0.0193339]]))=}"
)

clip_id = "munich2024-01-09-1080i-yadif"
frame_index = 8775
#jpg_path = Path(f"examples/{clip_id}_{frame_index:06d}_original.jpg").resolve()
#png_path = Path(f"examples/{clip_id}_{frame_index:06d}_original.png").resolve()
png_path = Path("examples/good.png").resolve()
jpg_path = Path("examples/good.jpg").resolve()

jpg = open_as_rgb_hwc_np_u8(
    jpg_path
)
png = open_as_rgb_hwc_np_u8(
    png_path
)

png_rgb_values = png.reshape(-1, 3).astype(np.float64)
jpg_rgb_values = jpg.reshape(-1, 3).astype(np.float64)

column_of_ones = np.ones(
    shape=(len(png_rgb_values), 1),
    dtype=np.float64
)

M = np.zeros(
    shape=(4, 3)
)
png_CIE_values = sRGB_to_CIE_linear_XYZ(png_rgb_values)
jpg_CIE_values = sRGB_to_CIE_linear_XYZ(jpg_rgb_values)

X = np.hstack(
        (column_of_ones, png_CIE_values)
)
for c in range(3):
    y = jpg_CIE_values[:, c]

    regr = linear_model.LinearRegression()
    regr.fit(X, y)

    coeffs = regr.coef_
    print(f"{coeffs=}")
    M[:, c] = coeffs

predicted_jpg_CIE_values = np.dot(X, M)
predicted_jpg_rgb_values = CIE_linear_XYZ_to_sRGB(predicted_jpg_CIE_values)

residuals = np.abs(predicted_jpg_rgb_values - jpg_rgb_values)
L_1_error = np.mean(residuals)
L_2_error = np.sqrt(np.mean(residuals**2))
L_infinity_error = np.max(residuals)
print(f"{L_1_error=}")
print(f"{L_2_error=}")
print(f"{L_infinity_error=}")

fixed_f64 = predicted_jpg_rgb_values.reshape(jpg.shape)
fixed_u8 = np.round(fixed_f64).clip(0, 255).astype(np.uint8)
fixed_path = Path(f"examples/{clip_id}_{frame_index:06d}_fixed.png").resolve()
write_rgb_hwc_np_u8_to_png(
    rgb_hwc_np_u8=fixed_u8,
    out_abs_file_path=fixed_path,
)
print("try:")
print(f"ff {fixed_path} {jpg_path}")







