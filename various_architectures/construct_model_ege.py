from egeunet import EGEUNet


def construct_model_ege(num_class = 2, include_classification = False):
    model = EGEUNet(num_classes=2)
    return model


