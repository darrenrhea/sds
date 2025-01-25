from smp_unet import SmpUnet

import segmentation_models_pytorch as smp


# efficientnet v2 based unet
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/decoders/unet/model.py
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/base/model.py
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/base/heads.py
# https://medium.com/aiguys/review-efficientnetv2-smaller-models-and-faster-training-47d4215dcdfb
# https://github.com/google/automl/blob/master/efficientnetv2/effnetv2_configs.py
# https://arxiv.org/abs/2104.00298
# https://arxiv.org/abs/2307.03980
# https://github.com/lostmartian/Building-and-Road-Segmentation-from-Aerial-Images
# https://github.com/lostmartian/Building-and-Road-Segmentation-from-Aerial-Images/blob/main/notebooks/road/V2S.ipynb



def construct_model_eff(model_size, in_channels=3, num_class = 2, include_classification = False, return_features = True, arch='u'):
    if include_classification:
        classification_head_params = {
            'classes': 5,
            'pooling': 'avg', # avg or max
            'dropout': 0.2, # dropout factor
            'activation': 'sigmoid', # sigmoid
        }
    else:
        classification_head_params = None

    encoder = 'tu-tf_efficientnetv2_' + model_size
    encoder_weights = 'imagenet'
    if arch == 'u':
        model = SmpUnet(
            in_channels = in_channels,
            encoder = encoder,
            encoder_weights = encoder_weights,
            encoder_depth = 5,
            decoder_use_batchnorm = True,
            decoder_channels = (256, 128, 64, 32, 16),
            classes = num_class,
            activation = "sigmoid", # put "sigmoid" here to bake logistic_sigmoid into effs, None for way training currently works.
            # attention results in fuzzy edges
            # decoder_attention_type = 'scse', # can be None, https://arxiv.org/abs/1808.08127
            decoder_attention_type = None,
            aux_params = classification_head_params,
            return_features = return_features,
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


