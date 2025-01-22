from do_polynomial_fit import (
     do_polynomial_fit
)

from matplotlib import pyplot as plt

from prii import (
     prii
)

import numpy as np



def correct_each_channel_independently(
    uncorrected: np.ndarray,  # shape=(n_rows, n_cols, 3)
    from_to_mapping_array,  # shape=(n_colors, 2, 3)
    channel_to_config: dict,
    graph_it=False,
):
   
    assert uncorrected.dtype == np.float64
    assert uncorrected.shape[2] == 3
    assert from_to_mapping_array.shape[1] == 2
    assert from_to_mapping_array.shape[2] == 3

    corrected = np.zeros_like(uncorrected)

    for channel_index in range(from_to_mapping_array.shape[2]):
        config = channel_to_config[channel_index]
        x_min = config["min"]
        x_max = config["max"]
        degree = config["degree"]

        xs = from_to_mapping_array[:, 0, channel_index]
        ys = from_to_mapping_array[:, 1, channel_index]
        uncorrected_channel = uncorrected[:, :, channel_index]

       
        model = do_polynomial_fit(xs=xs, ys=ys, degree=degree)

        corrected_channel_preclip = model(uncorrected_channel)
        corrected_channel = np.clip(
            corrected_channel_preclip,
            x_min,
            x_max
        )

        corrected[:, :, channel_index] = corrected_channel

      
        if graph_it:
            x_plot_values = np.linspace(x_min, x_max, 1000)
            y_plot_values = model(x_plot_values)
            y_plot_values = np.clip(y_plot_values, x_min, x_max)

            plt.figure()
            plt.plot(x_plot_values, y_plot_values, "-")
            plt.plot(xs, ys, "k.")
            plt.xlim(x_min, x_max)
            plt.xlabel("uncorrected")
            plt.ylabel("corrected")
            plt.title(f"{channel_index}")
            plot_image_file_path = f"{channel_index}.png"
            plt.savefig(plot_image_file_path)
            prii(plot_image_file_path)
      
    
  
    return corrected