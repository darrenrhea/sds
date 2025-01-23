from torchvision import transforms


def get_normalization_and_chw_transform():
    TRANSFORM_NORM_IMAGE = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ) # imagenet
    ])
    return TRANSFORM_NORM_IMAGE
