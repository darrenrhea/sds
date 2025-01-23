from smp_unet import SmpUnet

def construct_model_resnet34_based_unet(in_channels=3, num_class = 2):
    model = SmpUnet(
        encoder = "resnet34",
        encoder_depth = 5,
        encoder_weights = None,
        decoder_use_batchnorm = True,
        decoder_channels = (256, 128, 64, 32, 16),
        decoder_attention_type = None,
        in_channels = in_channels,
        classes = num_class,
        activation = None, # put "sigmoid" here to bake logistic_sigmoid into effs, None for way training currently works.
        # attention results in fuzzy edges
        # decoder_attention_type = 'scse', # can be None, https://arxiv.org/abs/1808.08127
        aux_params = None,
        return_features = False,
        scaleup=1
    )
    return model

