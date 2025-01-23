from typing import Optional, Union, List

import torch.nn as nn

from segmentation_models_pytorch.base.modules import Activation

from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
)

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

class ClassificationHead(nn.Sequential):
    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == "avg" else nn.AdaptiveMaxPool2d(1)
        flatten = nn.Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)

class ClassificationHeadPatches(nn.Module):
    def __init__(self, in_channels, classes, num_columns, num_rows, pooling="avg", dropout=0.2, activation=None):
        super().__init__()
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        self.pool = nn.AdaptiveAvgPool2d(1) if pooling == "avg" else nn.AdaptiveMaxPool2d(1)
        # concatenate all pooled patches into one vector
        self.dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        self.linear = nn.Linear(in_channels, classes, bias=True)
        self.activation = Activation(activation)
        self.num_columns = num_columns
        self.num_rows = num_rows

    def forward(self, x):
        # x dimensions: batch x feats x height x width
        x = self.pool(x)
        y = x.permute(1, 2, 3, 0)
        # y dimensions: feats x height x width x batch
        # z = x.reshape(y.shape[0], y.shape[1] * self.num_rows, y.shape[2] * self.num_columns).unsqueeze(0)
        # z dimensions: 1 x feats x height x width
        out1 = y.view(1, -1)
        #out2 = self.activation(self.linear(self.dropout(out1)))
        #return out2
        return out1

class ClassificationHeadPatchesTransformer(nn.Module):
    def __init__(self, num_columns, num_rows):
        super().__init__()
        self.num_columns = num_columns
        self.num_rows = num_rows
    
    def forward(self, x):
        # x dimensions: batch x feats x height x width
        y = x.permute(1, 2, 3, 0)
        # y dimensions: feats x height x width x batch
        z = x.reshape(y.shape[0], y.shape[1] * self.num_rows, y.shape[2] * self.num_columns).unsqueeze(0)
        # z dimensions: 1 x feats x height x width
        # TODO: put through transformer head
        # https://github.com/Alibaba-MIIL/ML_Decoder
        return z



class SmpSegmentationModel(SegmentationModel):

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        #print('smp features', len(features), [x.shape for x in features])
        
        decoder_output = self.decoder(*features)

        #print('smp decoder', decoder_output.shape)

        if self.scaleup > 1:
            decoder_output = nn.functional.interpolate(decoder_output, scale_factor = 2, mode = 'bilinear', align_corners=False)
        #print('features', len(features), list(map(lambda x: x.shape, features)))

        #print('smp decoder', decoder_output.shape)

        masks = self.segmentation_head(decoder_output)

        # can have several other heads here, too..
        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        if self.return_features:
            return masks, features[-1]

        return masks

# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/decoders/unet/model.py
class SmpUnet(SmpSegmentationModel):
    def __init__(
        self,
        encoder = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        return_features = False,
        scaleup = 1
    ):
        self.scaleup = scaleup
        super().__init__()

        if isinstance(encoder, str):
            self.encoder = get_encoder(
                encoder,
                in_channels=in_channels,
                depth=encoder_depth,
                weights=encoder_weights,
            )
            center=True if encoder.startswith("vgg") else False
        else:
            self.encoder = encoder
            center = False
            encoder = str(encoder)
        

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=center,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.return_features = return_features

        self.name = "u-{}".format(encoder)
        self.initialize()
