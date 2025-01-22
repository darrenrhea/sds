# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
import sys
import numpy as np
from scipy.optimize import least_squares
import better_json as bj
import PIL.Image
from pathlib import Path
from valid_context_ids import valid_context_ids
from colorama import Fore, Style

def use_x_ie_the_values_of_the_parameters_to_make_predictions(x, domain_rgb_triplets):
    """
    This is where the ansatz/functional form is defined.
    Currently the ansatz is: affine function.
    given fixed values x of the parameters,
    apply it to a bunch of rgb triplets to make predictions
    """
    assert isinstance(x, np.ndarray)
    assert x.ndim == 1

    num_data_points = domain_rgb_triplets.shape[0]

    design_matrix = np.ones(
        shape=(num_data_points, 4)
    )

    design_matrix[:, 0:3] = domain_rgb_triplets

    y = np.dot(design_matrix, x)

    return y

def f(x, *args, **kwargs):
    """
    we optimize over x
    """
    assert isinstance(x, np.ndarray)
    assert x.ndim == 1

    actual_values = kwargs["actual_values"]
    domain_rgb_triplets =  kwargs["domain_rgb_triplets"]
    num_data_points = domain_rgb_triplets.shape[0]

    design_matrix = np.ones(
        shape=(num_data_points, 4)
    )

    design_matrix[:, 0:3] = domain_rgb_triplets

    predicted_values = use_x_ie_the_values_of_the_parameters_to_make_predictions(
        x=x,
        domain_rgb_triplets=domain_rgb_triplets
    )

    y = predicted_values - actual_values

    assert y.shape == (num_data_points, )
    return y





def determine_channel_function(color_map, channel: str):
    actual_values_list = []
    list_of_rgb_triplets = []
    for dct in color_map:
        domain = dct["domain"]
        range = dct["range"]
        actual_value = range[f"{channel}_average"]
        actual_values_list.append(actual_value)
        rgb_triplet = [
            domain["r_average"],
            domain["g_average"],
            domain["b_average"],
        ]
        list_of_rgb_triplets.append(rgb_triplet)
    
    actual_values = np.array(
        actual_values_list,
        dtype=np.float64
    )

    domain_rgb_triplets = np.array(
        list_of_rgb_triplets,
        dtype=np.float64
    )

    print(f"actual_values=")
    print(actual_values)
    print(f"domain_rgb_triplets=")
    print(domain_rgb_triplets)

    x0 = np.array([1, 1, 1, 0], dtype=np.float64)

    constants = dict(
        actual_values=actual_values,
        domain_rgb_triplets=domain_rgb_triplets
    )

    result = least_squares(
        fun=f,
        x0=x0,
        jac='2-point',
        bounds=(-np.inf, np.inf),
        method='trf',
        ftol=1e-08,
        xtol=1e-08,
        gtol=1e-08,
        x_scale=1.0,
        loss='linear',
        f_scale=1.0,
        diff_step=None,
        tr_solver=None,
        tr_options={},
        jac_sparsity=None,
        max_nfev=None,
        verbose=1,
        args=(),
        kwargs=constants
    )

    x_star = result["x"]
    print(f"{x_star=}")

    predictions = use_x_ie_the_values_of_the_parameters_to_make_predictions(
        x=x_star,
        domain_rgb_triplets=domain_rgb_triplets
    )

    print(f"For channel {channel}:")

    display = np.stack(
        [predictions, actual_values],
        axis=1
    )

    print(f"{display}")
    return x_star


def main():
    context_id = sys.argv[1]
    assert context_id in valid_context_ids

    color_map = bj.load(f"~/needs_color_correction/{context_id}/{context_id}_color_map.json")

    channel_to_x_star = dict()

    for channel in ["r", "g", "b"]:
        channel_to_x_star[channel] = determine_channel_function(
            channel=channel,
            color_map=color_map
        )

    image_file_path = Path(
        f"~/needs_color_correction/{context_id}/{context_id}_ripped.png"
    ).expanduser()

                               
    if not image_file_path.exists():
        print(f"{Fore.RED}ERROR: {image_file_path} does not exist!{Style.RESET_ALL}")
        sys.exit(1)

    image_pil = PIL.Image.open(str(image_file_path))
    image_np = np.array(image_pil)[:, :, :3].astype(np.float64)  # screw alpha channel
    as_rgb_triplets = image_np.reshape((-1, 3))
    
    print(as_rgb_triplets.shape)
    
    predicted = np.zeros(
        shape=as_rgb_triplets.shape,
        dtype=np.float64
    )

    for i, channel in enumerate(["r", "g", "b"]):
        x_star = channel_to_x_star[channel]
        predicted[:, i] = use_x_ie_the_values_of_the_parameters_to_make_predictions(
            x=x_star,
            domain_rgb_triplets=as_rgb_triplets
        )

    predicted_np_uint8 = predicted.clip(0, 255).astype(np.uint8).reshape(image_np.shape)
    
    predicted_pil = PIL.Image.fromarray(predicted_np_uint8)
    predicted_pil.save("out.png")

    print(f"See out.png")

    
    
if __name__ == "__main__":
    main()
