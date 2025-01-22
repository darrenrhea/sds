from get_unique_rgb_combinations_with_counts import (
     get_unique_rgb_combinations_with_counts
)
import numpy as np
import sklearn
import sklearn.cluster
import pandas as pd
#Starting k-means clustering


def quantize_colors_via_kmeans(
    rgb_values: np.ndarray,
    num_colors_to_quantize_to: int
):
    """
    Suppose you have a bunch of rgb values in rgbs,
    a 3 column numpy uint8 array.
    This uses k-means clustering to find the 3 most common colors in rgbs,
    and returns
    the cluster indices and the centroids that were found
    rounded to the nearest integer.
    """
    assert isinstance(rgb_values, np.ndarray)
    assert rgb_values.dtype == np.uint8
    assert rgb_values.ndim == 2
    assert rgb_values.shape[1] == 3

    unique_rgb_values, counts = get_unique_rgb_combinations_with_counts(
        rgb_values
    )

    X = unique_rgb_values
    sample_weight = counts

    centroids = np.array(
        [
            [255,0,0],
            [0,255,0],
            [0,0,255]
        ]
    )

    kmeans = sklearn.cluster.KMeans(
        init="random",
        n_clusters=num_colors_to_quantize_to,
        n_init=100,
        random_state=0,
        max_iter=1000
    )

    # Running k-means clustering and enter the ‘X’ array as the input coordinates and ‘Y’  array as sample weights
    _ = kmeans.fit(X, sample_weight = sample_weight)
    cluster_indices = kmeans.predict(rgb_values)


    # Displaying the cluster centers
    centroids_u8 = np.round(kmeans.cluster_centers_).astype(np.uint8)

    # trouble is, they may not be sorted:
    _, counts = np.unique(cluster_indices, return_counts=True)

    tuples = [(i, counts[i]) for i in range(len(counts))]
    tuples.sort(key=lambda x: x[1], reverse=True)

    current_indices_in_descending_frequency = [t[0] for t in tuples]
    current_index_new_index_pairs = [
        (current_indices_in_descending_frequency[i], i)
        for i in range(len(current_indices_in_descending_frequency))
    ]
    current_index_new_index_pairs.sort(key=lambda x: x[0])

    current_index_to_new_index = np.array(
        [
            pair[1]
            for pair in current_index_new_index_pairs
        ]
    )

    new_cluster_indices = current_index_to_new_index[cluster_indices]

    new_centroids_u8 = centroids_u8[current_indices_in_descending_frequency]

    return new_cluster_indices, new_centroids_u8
