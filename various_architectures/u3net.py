# https://github.com/dmMaze/UNet3Plus-pytorch

# TODO: add normalization
#

import sys, os
import numpy as np
from typing import List
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import torchvision
from torchvision.models import resnet18, resnet34, resnet50, resnet101, ResNet
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

import timm

from segmentation_models_pytorch.encoders import get_encoder


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        weights_init_kaiming(m)
    elif isinstance(m, nn.BatchNorm2d):
        weights_init_kaiming(m)

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def u3pblock(in_ch, out_ch, num_block=2, kernel_size=3, padding=1, down_sample=False):
    m = []
    if down_sample:
        m.append(nn.MaxPool2d(kernel_size=2))
    for _ in range(num_block):
        m += [nn.Conv2d(in_ch, out_ch, kernel_size, bias=False, padding=padding),
              nn.BatchNorm2d(out_ch),
              nn.ReLU(inplace=True)]
        in_ch = out_ch
    return nn.Sequential(*m)

def en2dec_layer(in_ch, out_ch, scale):
    m = [nn.Identity()] if scale == 1 else [nn.MaxPool2d(scale, scale, ceil_mode=True)]
    m.append(u3pblock(in_ch, out_ch, num_block=1))
    return nn.Sequential(*m)

def dec2dec_layer(in_ch, out_ch, scale, fast_up=True):
    up = [nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True) if scale != 1 else nn.Identity()]
    m = [u3pblock(in_ch, out_ch, num_block=1)]
    if fast_up:
        m = m + up
    else:
        m = up + m  # used in paper
    return nn.Sequential(*m)


class FullScaleSkipConnect(nn.Module):
    def __init__(self,
                 en_channels,   # encoder out channels, high to low
                 en_scales,
                 num_dec,       # number of decoder out
                 skip_ch=64,
                 dec_scales=None,
                 bottom_dec_ch=1024,
                 dropout=0.3,
                 fast_up=True,):

        super().__init__()
        concat_ch = skip_ch * (len(en_channels) + num_dec)

        # encoder maps to decoder maps connections
        self.en2dec_layers = nn.ModuleList()
        # print(en_scales)
        for ch, scale in zip(en_channels, en_scales):
            self.en2dec_layers.append(en2dec_layer(ch, skip_ch, scale))

        # decoder maps to decoder maps connections
        self.dec2dec_layers = nn.ModuleList()
        if dec_scales is None:
            dec_scales = []
            for ii in reversed(range(num_dec)):
                dec_scales.append(2 ** (ii + 1))
        for ii, scale in enumerate(dec_scales):
            dec_ch = bottom_dec_ch if ii == 0 else concat_ch
            self.dec2dec_layers.append(dec2dec_layer(dec_ch, skip_ch, scale, fast_up=fast_up))

        self.droupout = nn.Dropout(dropout)
        self.fuse_layer = u3pblock(concat_ch, concat_ch, 1)

    def forward(self, en_maps, dec_maps=None):
        out = []
        for en_map, layer in zip(en_maps, self.en2dec_layers):
            out.append(layer(en_map))
        if dec_maps is not None and len(dec_maps) > 0:
            for dec_map, layer in zip(dec_maps, self.dec2dec_layers):
                out.append(layer(dec_map))
        return self.fuse_layer(self.droupout(torch.cat(out, 1)))


class U3PEncoderDefault(nn.Module):
    def __init__(self, channels = [3, 64, 128, 256, 512, 1024], num_block=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.downsample_list = nn.Module()
        for ii, (ch_in, ch_out) in enumerate(zip(channels[:-1], channels[1:])):
            self.layers.append(u3pblock(ch_in, ch_out, num_block, down_sample= ii > 0))
        self.channels = channels
        self.apply(weight_init)

    def forward(self, x):
        encoder_out = []
        for layer in self.layers:
            x = layer(x)
            encoder_out.append(x)
        return encoder_out

class U3PEncoderEffNetV2(nn.Module):
    def __init__(self, model_size):
        super(U3PEncoderEffNetV2, self).__init__()
        self.model = get_encoder(
            f'tu-tf_efficientnetv2_{model_size}',
            in_channels=3,
            depth=5,
            weights='imagenet',
        )
        self.channels = self.model.out_channels
        #self.model = timm.create_model('tu-tf_efficientnetv2_s',pretrained=True,features_only=True)
        #self.channels = self.model.feature_info.channels()

    def forward(self, x):
       return self.model.forward(x)



class U3PDecoder(nn.Module):
    def __init__(self, en_channels = [64, 128, 256, 512, 1024], skip_ch=64, dropout=0.3, fast_up=True):
        super().__init__()
        self.decoders = nn.ModuleDict()
        en_channels = en_channels[::-1]
        num_en_ch = len(en_channels)
        for ii in range(num_en_ch):
            if ii == 0:
                # first decoding output is identity mapping of last encoder map
                self.decoders['decoder1'] = nn.Identity()
                continue

            self.decoders[f'decoder{ii+1}'] = FullScaleSkipConnect(
                                                en_channels[ii:],
                                                en_scales=2 ** np.arange(0, num_en_ch-ii),
                                                num_dec=ii,
                                                skip_ch=skip_ch,
                                                bottom_dec_ch=en_channels[0],
                                                dropout=dropout,
                                                fast_up=fast_up
                                            )

    def forward(self, enc_map_list:List[torch.Tensor]):
        dec_map_list = []
        enc_map_list = enc_map_list[::-1]
        for ii, layer_key in enumerate(self.decoders):
            layer = self.decoders[layer_key]
            if ii == 0:
                dec_map_list.append(layer(enc_map_list[0]))
            else:
                dec_map_list.append(layer(enc_map_list[ii: ], dec_map_list))
        return dec_map_list



class UNet3Plus(nn.Module):

    def __init__(self,
                 num_classes=1,
                 skip_ch=64,
                 aux_losses=2,
                 encoder = None,
                 channels=[3, 64, 128, 256, 512, 1024],
                 dropout=0.3,
                 transpose_final=False,
                 use_cgm=True,
                 fast_up=True,
                 include_sigmoid=False  # during training needs to be False until we remove the sigmoid from the loss function calculation
        ):
        super().__init__()

        self.include_sigmoid = include_sigmoid

        self.encoder = U3PEncoderDefault(channels) if encoder is None else encoder
        channels = self.encoder.channels
        num_decoders = len(channels) - 1
        decoder_ch = skip_ch * num_decoders

        self.decoder = U3PDecoder(self.encoder.channels[1:], skip_ch=skip_ch, dropout=dropout, fast_up=fast_up)
        self.decoder.apply(weight_init)

        self.cls = nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Conv2d(channels[-1], 2, 1),
                    nn.AdaptiveMaxPool2d(1),
                    nn.Sigmoid()
                ) if use_cgm and num_classes <= 2 else None

        if transpose_final:
            self.head = nn.Sequential(
                nn.ConvTranspose2d(decoder_ch, num_classes, kernel_size=4, stride = 2, padding=1, bias=False),
            )
        else:
            self.head = nn.Conv2d(decoder_ch, num_classes, 3, padding=1)
        self.head.apply(weight_init)

        if aux_losses > 0:
            self.aux_head = nn.ModuleDict()
            layer_indices = np.arange(num_decoders - aux_losses - 1, num_decoders - 1)
            for ii in layer_indices:
                ch = decoder_ch if ii != 0 else channels[-1]
                self.aux_head.add_module(f'aux_head{ii}', nn.Conv2d(ch, num_classes, 3, padding=1))
            self.aux_head.apply(weight_init)
        else:
            self.aux_head = None

    def forward(self, x):
        _, _, h, w = x.shape
        de_out = self.decoder(self.encoder(x))
        have_obj = 1

        pred = self.resize(self.head(de_out[-1]), h, w)

        if self.training:
            pred = {'final_pred': pred}
            if self.aux_head is not None:
                for ii, de in enumerate(de_out[:-1]):
                    if ii == 0:
                        if self.cls is not None:
                            pred['cls'] = self.cls(de).squeeze_()
                            have_obj = torch.argmax(pred['cls'])
                    head_key = f'aux_head{ii}'
                    if head_key in self.aux_head:
                        de: torch.Tensor = de * have_obj
                        pred[f'aux{ii}'] = self.resize(self.aux_head[head_key](de), h, w)
        
        if self.include_sigmoid:
            print('NOTE:SIGMOID APPLIED')
            pred = torch.sigmoid(pred)
        
        return pred

    def resize(self, x, h, w) -> torch.Tensor:
        _, _, xh, xw = x.shape
        if xh != h or xw != w:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x


resnets = ['resnet18', 'resnet34', 'resnet50', 'resnet101']
efficientnets = ['effs', 'effm', 'effl']
convnexts = ['convnextv2t', 'convnextv2s', 'convnextv2b', 'convnexts', 'convnextb']
fasternets = ['fasternets', 'fasternetm', 'fasternetl']

encoder_configs = {
    'return_nodes_res': {
        'relu': 'layer0',
        'layer1': 'layer1',
        'layer2': 'layer2',
        'layer3': 'layer3',
        'layer4': 'layer4',
    },
    'resnet18': {
        'fe_channels': [64, 64, 128, 256, 512],
        'channels': [32, 64, 128, 256, 512],
    },
    'resnet50': {
        'fe_channels': [64, 256, 512, 1024, 2048],
        'channels': [64, 128, 256, 512, 1024],
    },
    'effs': {
        'fe_channels': [24, 48, 64, 160, 256],
        'channels': [24, 48, 64, 160, 256],
        #'channels': [32, 64, 128, 256, 512], # minimally slower on RTX 6000
    },
    'effm': {
        'fe_channels': [24, 48, 80, 176, 512],
        'channels': [24, 48, 80, 176, 512],
    },
    'convnextv2t': {
        'fe_channels': [96, 192, 384, 768],
        'channels': [96, 192, 384, 768],
    },
    'convnextv2s': {
        'fe_channels': [96, 192, 384, 768],
        'channels': [96, 192, 384, 768],
    },
    'convnextv2b': {
        'fe_channels': [128, 256, 512, 1024],
        'channels': [128, 256, 512, 1024],
    },
    'convnexts': {
        'fe_channels': [96, 192, 384, 768],
        'channels': [96, 192, 384, 768],
    },
    'convnextb': {
        'fe_channels': [128, 256, 512, 1024],
        'channels': [128, 256, 512, 1024],
    },
    'fasternets': {
        'fe_channels': [128, 256, 512, 1024],
        'channels': [128, 256, 512, 1024],
    },
    'fasternetm': {
        'fe_channels': [144, 288, 576, 1152],
        'channels': [144, 288, 576, 1152],
    },
    'fasternetl': {
        'fe_channels': [192, 384, 768, 1536],
        'channels': [192, 384, 768, 1536],
    },
}



class U3PResNetEncoder(nn.Module):
    '''
    ResNet encoder wrapper
    '''
    def __init__(self, backbone='resnet18', pretrained=False) -> None:
        super().__init__()
        resnet: ResNet = globals()[backbone](pretrained=pretrained)
        cfg = encoder_configs['resnet18'] if backbone in ['resnet18', 'resnet34'] else encoder_configs['resnet50']
        if not pretrained:
            resnet.apply(weight_init)

        self.backbone = create_feature_extractor(resnet, return_nodes=encoder_configs['return_nodes_res'])

        # print(resnet)
        # input = torch.randn(1, 3, 320, 320)
        # out = self.backbone(input)

        self.compress_convs = nn.ModuleList()
        for ii, (fe_ch, ch) in enumerate(zip(cfg['fe_channels'], cfg['channels'])):
            if fe_ch != ch:
                self.compress_convs.append(nn.Conv2d(fe_ch, ch, 1, bias=False))
            else:
                self.compress_convs.append(nn.Identity())

        self.channels = [3] + cfg['channels']

    def forward(self, x):
        out = self.backbone(x)
        for ii, compress in enumerate(self.compress_convs):
            out[f'layer{ii}'] = compress(out[f'layer{ii}'])
        out = [v for _, v in out.items()]
        return out


class U3PEffNetEncoder(nn.Module):
    '''
    EfficietNet encoder wrapper
    '''
    def __init__(self, backbone='effs', pretrained=True) -> None:
        super().__init__()

        model_size = backbone[-1]

        model = get_encoder(
            f'tu-tf_efficientnetv2_{model_size}',
            in_channels=3,
            depth=5,
            weights='imagenet' if pretrained else None,
        )
        self.backbone = model

        print(model.out_channels)

        if not pretrained:
            model.apply(weight_init)

        cfg = encoder_configs[backbone]

        self.compress_convs = nn.ModuleList()
        for ii, (fe_ch, ch) in enumerate(zip(cfg['fe_channels'], cfg['channels'])):
            if fe_ch != ch:
                self.compress_convs.append(nn.Conv2d(fe_ch, ch, 1, bias=False))
            else:
                self.compress_convs.append(nn.Identity())

        self.channels = [3] + cfg['channels']

    def forward(self, x):
        out = self.backbone(x)
        compressed = [comp(l) for comp, l in zip(self.compress_convs, out[1:])]
        return compressed

class U3PConvNextEncoder(nn.Module):
    '''
    ConvNextV2 encoder wrapper
    '''
    def __init__(self, backbone='convnextv2s', pretrained=True) -> None:
        super().__init__()

        model_size = backbone[-1]
        MODEL_NAMES = {
            't': 'tiny',
            's': 'small',
            'b': 'base',
        }
        if 'v2' in backbone:
            model_name = f'convnextv2_{MODEL_NAMES[model_size]}'
            pretrained = 'fcmae_ft_in1k' if model_size != 's' else False
        else:
            model_name = f'convnext_{MODEL_NAMES[model_size]}'
            pretrained = 'in12k_ft_in1k_384'

        model = timm.create_model(model_name, pretrained=pretrained, features_only=True)

        data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        print(data_cfg)

        transform = torchvision.transforms.Normalize(mean = data_cfg['mean'], std = data_cfg['std'])
        print(transform)

        self.backbone = model

        print(model.feature_info.channels())

        if not pretrained:
            model.apply(weight_init)

        cfg = encoder_configs[backbone]

        self.compress_convs = nn.ModuleList()
        for ii, (fe_ch, ch) in enumerate(zip(cfg['fe_channels'], cfg['channels'])):
            if fe_ch != ch:
                self.compress_convs.append(nn.Conv2d(fe_ch, ch, 1, bias=False))
            else:
                self.compress_convs.append(nn.Identity())

        self.channels = [3] + cfg['channels']

    def forward(self, x):
        out = self.backbone(x)
        #compressed = [comp(l) for comp, l in zip(self.compress_convs, out[1:])]
        #return compressed
        return out

class U3PFasterNetEncoder(nn.Module):
    '''
    FasterNet encoder wrapper
    '''
    def __init__(self, backbone='fasternets', pretrained=True) -> None:
        super().__init__()
        from fasternet import fasternet_s, fasternet_m, fasternet_l
        model_size = backbone[-1]
        MODEL_NAMES = {
            's': fasternet_s,
            'm': fasternet_m,
            'l': fasternet_l,
        }

        model = MODEL_NAMES[model_size](in_chans=3, num_classes=1)
        # TODO: normalization

        self.backbone = model

        print('fasternet pretrained', pretrained)
        if pretrained:
            CHECKPOINTS = {
                's': 'pretrained/fasternet_s-epoch.299-val_acc1.81.2840.pth',
                'm': 'pretrained/fasternet_m-epoch.291-val_acc1.82.9620.pth',
                'l': 'pretrained/fasternet_l-epoch.299-val_acc1.83.5060.pth',
            }
            
            ckpt = torch.load(CHECKPOINTS[model_size], map_location = 'cpu')
            
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                model.load_state_dict(state_dict, strict=False)

            print('missing_keys: ', missing_keys)
            print('unexpected_keys: ', unexpected_keys)

        else:
            model.apply(weight_init)

        cfg = encoder_configs[backbone]

        self.compress_convs = nn.ModuleList()
        for ii, (fe_ch, ch) in enumerate(zip(cfg['fe_channels'], cfg['channels'])):
            if fe_ch != ch:
                self.compress_convs.append(nn.Conv2d(fe_ch, ch, 1, bias=False))
            else:
                self.compress_convs.append(nn.Identity())

        self.channels = [3] + cfg['channels']

    def forward(self, x):
        out = self.backbone(x)
        #compressed = [comp(l) for comp, l in zip(self.compress_convs, out[1:])]
        #return compressed
        return out



def build_unet3plus(num_classes, encoder='default', skip_ch=64, aux_losses=2, use_cgm=False, pretrained=True, dropout=0.2) -> UNet3Plus:
    if encoder == 'default':
        encoder = None
        aux_losses = 4
        dropout = 0.0
        transpose_final = False
        fast_up = False
    elif encoder in resnets:
        encoder = U3PResNetEncoder(backbone=encoder, pretrained=pretrained)
        transpose_final = True
        fast_up = True
    elif encoder in efficientnets:
        encoder = U3PEffNetEncoder(backbone=encoder, pretrained=pretrained)
        transpose_final = True
        fast_up = True
    elif encoder in convnexts:
        encoder = U3PConvNextEncoder(backbone=encoder, pretrained=pretrained)
        transpose_final = True
        fast_up = True
    elif encoder in fasternets:
        encoder = U3PFasterNetEncoder(backbone=encoder, pretrained=pretrained)
        transpose_final = True
        fast_up = True
    else:
        raise ValueError(f'Unsupported backbone : {encoder}')
   
    model = UNet3Plus(
        num_classes=num_classes,
        skip_ch=skip_ch,
        aux_losses=aux_losses,
        encoder=encoder,
        channels=[3, 64, 128, 256, 512, 1024],
        use_cgm=use_cgm,
        dropout=dropout,
        transpose_final=transpose_final,
        fast_up=fast_up,
        include_sigmoid=False  # needs to be False until we change the way the loss is calculated
    )
    return model
