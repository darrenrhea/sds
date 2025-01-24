from u3net import build_unet3plus


def construct_model_u3(backbone, in_channels=3, num_class = 2, include_classification = False, return_features = False):

    model = build_unet3plus(
        num_classes=num_class,
        encoder=backbone,
        pretrained=False,  # Does not seem like the pretrained models are happy
    )
    def preprocessing_fn(x):
        return x

    return model, preprocessing_fn

