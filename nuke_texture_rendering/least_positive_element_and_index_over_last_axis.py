import numpy as np

def least_positive_element_and_index_over_last_axis(array):
    """
    If there are no positive elements in a row, the minimum positive element is set to infinity.
    """
    # Find indices of positive elements
    positive_indices = np.where(array > 0, array, np.inf)

    # Find the minimum positive element along each row
    min_positive_values = np.min(positive_indices, axis=-1)

    # Find the column index of the minimum positive element along each row
    min_positive_indices = np.argmin(positive_indices, axis=-1)

    assert min_positive_values.shape == array.shape[:-1]
    assert min_positive_indices.shape == array.shape[:-1]
    return min_positive_values, min_positive_indices

if __name__ == "__main__":
        
    # Example usage:
    arr = np.array([
        [
            [5, -1, 4],
            [9,  3, 4]
        ],

        [
            [7, 3, 2],
            [-2, 4, 7]
        ],

        [
            [1, 3, 0],
            [8, 3, 4]
        ],
    ])

    min_values, min_indices = least_positive_element_and_index_over_last_axis(arr)

    print("Least positive element of each row:")
    print(min_values)
    print("index where it occurs:")
    print(min_indices)