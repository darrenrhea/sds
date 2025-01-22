
import torch
import numpy as np
from randomization_utils import choose_name_from_name_to_prob


class SegmentationDataset(torch.utils.data.IterableDataset):
    """
    this has relevance concept baked in.
    """
    def __init__(
        self,
        nn_input_height,
        nn_input_width,
        urns_with_probs_and_data_augmenters,
        mask_name_to_predict,
        relevance_mask_name
    ):
        """
        """
        self.urns_with_probs_and_data_augmenters = urns_with_probs_and_data_augmenters
        self.mask_name_to_predict = mask_name_to_predict
        self.relevance_mask_name = relevance_mask_name
        self.crop_height = urns_with_probs_and_data_augmenters["crop_height"]
        self.crop_width = urns_with_probs_and_data_augmenters["crop_width"]
        self.nn_input_height = nn_input_height
        self.nn_input_width = nn_input_width

        # needed to normalize color channels
        self.mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def generator(self):
        while True:
            # pick an urn / datasource according to the stated urn probabilities
            urn_name = choose_name_from_name_to_prob(
                self.urns_with_probs_and_data_augmenters["urn_name_to_probability_of_sampling_from_that_urn"]
            )
            urn = self.urns_with_probs_and_data_augmenters["urn_name_to_urn"][urn_name]

            # get that urn's augmenter
            data_augmenter = self.urns_with_probs_and_data_augmenters["urn_name_to_augmenter"][urn_name]

            image_index = np.random.randint(0, urn["num_croppings_cut"])  # draw at random from that urn
            x = urn["cropped_originals"][image_index, :, :, :]
            augmented_np = data_augmenter(x)  # data augmentation such as blurring should happen on the slightly bigger one

            di = np.random.randint(0, self.crop_height - self.nn_input_height + 1)
            dj = np.random.randint(0, self.crop_width - self.nn_input_width + 1)
            slightly_smaller_np = augmented_np[di:di + self.nn_input_height, dj:dj + self.nn_input_width, :]

            x_float32 = slightly_smaller_np.astype(np.float32)

            y = urn["mask_name_to_cropped_masks"][self.mask_name_to_predict][
                image_index,
                di:di + self.nn_input_height,
                dj:dj + self.nn_input_width
            ].astype(np.int64)  # it likes only int64

            z = urn["mask_name_to_cropped_masks"][self.relevance_mask_name][
                image_index,
                di:di + self.nn_input_height,
                dj:dj + self.nn_input_width
            ].astype(np.int64)  # it likes only int64

            standardized = (x_float32 / 255 - self.mean) / self.std
            standardized_chw = standardized.transpose([2, 0, 1])

            yield (
                torch.Tensor(standardized_chw),
                torch.LongTensor(y),
                torch.LongTensor(z)
            )  # would it be legal / good to return numpy instead of torch.Tensors?

    def __iter__(self):
        return self.generator()
