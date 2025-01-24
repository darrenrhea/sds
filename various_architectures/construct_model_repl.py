from smp_unet import SmpUnet
from replknet import RepLKNetEncoder


def construct_model_repl(model_size, in_channels=3, num_class = 2, include_classification = False, return_features = True):
    
    if include_classification:
        classification_head_params = {
            'classes': 5,
            'pooling': 'avg', # avg or max
            'dropout': 0.2, # dropout factor
            'activation': 'sigmoid', # sigmoid
        }
    else:
        classification_head_params = None

    include_stem = True
    backbone = RepLKNetEncoder(model_size, in_channels=in_channels, include_stem = include_stem)
    
    model = SmpUnet(
        in_channels = in_channels,
        encoder = backbone,
        encoder_depth = 5 if include_stem else 4,
        decoder_use_batchnorm = True,
        decoder_channels = [256, 128, 64, 32] + ([16] if include_stem else []),
        classes = num_class,
        activation = None, # 'sigmoid' - sigmoid does NOT work
        # attention results in fuzzy edges
        # decoder_attention_type = 'scse', # can be None, https://arxiv.org/abs/1808.08127
        decoder_attention_type = None,
        aux_params = classification_head_params,
        return_features = return_features,
        scaleup = 1 if include_stem else 2
    )

    #preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    preprocessing_fn = None

    return model, preprocessing_fn

