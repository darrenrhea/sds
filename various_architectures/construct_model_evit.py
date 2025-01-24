import segmentation_models_pytorch as smp

from smp_unet import SmpUnet


def construct_model_evit(model_size, in_channels=3, num_class = 2, include_classification = False, return_features = True, arch='u'):
    # if include_classification:
    #     classification_head_params = {
    #         'classes': 5,
    #         'pooling': 'avg', # avg or max
    #         'dropout': 0.2, # dropout factor
    #         'activation': 'sigmoid', # sigmoid
    #     }
    # else:
    #     classification_head_params = None

    classification_head_params = None

    encoder = 'tu-efficientvit_' + model_size
    encoder_weights = 'imagenet'

    if arch == 'u':
        model = SmpUnet(
            in_channels = in_channels,
            encoder = encoder,
            encoder_weights = encoder_weights,
            encoder_depth = 4,
            decoder_use_batchnorm = True,
            decoder_channels = (256, 128, 64, 32),
            classes = num_class,
            activation = None, # 'sigmoid' - sigmoid does NOT work
            # attention results in fuzzy edges
            # decoder_attention_type = 'scse', # can be None, https://arxiv.org/abs/1808.08127
            decoder_attention_type = None,
            aux_params = classification_head_params,
            return_features = return_features,
            scaleup = 2
        )
    elif arch == 'u++':
        model = smp.UnetPlusPlus(
            in_channels = in_channels,
            encoder_name = encoder,
            encoder_weights = encoder_weights,
            encoder_depth = 5,
            decoder_use_batchnorm = True,
            decoder_channels = (256, 128, 64, 32, 16),
            classes = num_class,
            activation = None, # 'sigmoid' - sigmoid does NOT work
            # attention results in fuzzy edges
            # decoder_attention_type = 'scse', # can be None, https://arxiv.org/abs/1808.08127
            decoder_attention_type = None,
            aux_params = classification_head_params,
            #return_features = return_features,
        )
    elif arch == 'ma':
        model = smp.MAnet(
            in_channels = in_channels,
            encoder_name = encoder,
            encoder_weights = encoder_weights,
            encoder_depth = 5,
            decoder_use_batchnorm = True,
            decoder_channels = (256, 128, 64, 32, 16),
            decoder_pab_channels=64,
            classes = num_class,
            activation = None, # 'sigmoid' - sigmoid does NOT work
            # attention results in fuzzy edges
            # decoder_attention_type = 'scse', # can be None, https://arxiv.org/abs/1808.08127
            #decoder_attention_type = None,
            aux_params = classification_head_params,
            #return_features = return_features,
        )
    else:
        raise NotImplementedError('unknown architecture')

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    return model, preprocessing_fn

