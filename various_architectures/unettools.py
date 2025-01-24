from load_model_u3 import (
     load_model_u3
)
from load_model_repl import (
     load_model_repl
)
from load_model_duat import (
     load_model_duat
)
from load_model_ege import (
     load_model_ege
)
from load_model_eff import (
     load_model_eff
)
from load_model2 import (
     load_model2
)
from load_model1 import (
     load_model1
)
from load_model_alwaysblack import (
     load_model_alwaysblack
)
from load_resnet34_based_unet import (
     load_resnet34_based_unet
)
from load_model_evit import (
     load_model_evit 
)

MODEL_LOADERS = {
    'alwaysblack': lambda *args, **kwargs: load_model_alwaysblack(*args, **kwargs),
    'res1': load_model1,
    'res2': load_model2,
    'effs': lambda *args, **kwargs: load_model_eff('s', *args, arch='u', **kwargs),
    'effsp': lambda *args, **kwargs: load_model_eff('s', *args, arch='u++', **kwargs),
    'effsma': lambda *args, **kwargs: load_model_eff('s', *args, arch='ma', **kwargs),
    'effm': lambda *args, **kwargs: load_model_eff('m', *args, **kwargs),
    'effl': lambda *args, **kwargs: load_model_eff('l', *args, **kwargs),
    'ege': load_model_ege,
    'duat': load_model_duat,
    'resnet34basedunet': load_resnet34_based_unet,
    'replb': lambda *args, **kwargs: load_model_repl('b', *args, **kwargs),
    'repll': lambda *args, **kwargs: load_model_repl('l', *args, **kwargs),
    'evitb0': lambda *args, **kwargs: load_model_evit('b0', arch='u', *args, **kwargs),
    'evitb1': lambda *args, **kwargs: load_model_evit('b1', arch='u', *args, **kwargs),
    'evitb2': lambda *args, **kwargs: load_model_evit('b2', arch='u', *args, **kwargs),
    'evitb3': lambda *args, **kwargs: load_model_evit('b3', arch='u', *args, **kwargs),
    'u3effs': lambda *args, **kwargs: load_model_u3('effs', *args, **kwargs),
    'u3effm': lambda *args, **kwargs: load_model_u3('effs', *args, **kwargs),
    'u3res34': lambda *args, **kwargs: load_model_u3('resnet34', *args, **kwargs),
    'u3res50': lambda *args, **kwargs: load_model_u3('resnet50', *args, **kwargs),
    'u3res101': lambda *args, **kwargs: load_model_u3('resnet101', *args, **kwargs),
    'u3convnexts': lambda *args, **kwargs: load_model_u3('convnexts', *args, **kwargs),
    'u3convnextm': lambda *args, **kwargs: load_model_u3('convnextm', *args, **kwargs),
    'u3fasternets': lambda *args, **kwargs: load_model_u3('fasternets', *args, **kwargs),
    'u3fasternetm': lambda *args, **kwargs: load_model_u3('fasternetm', *args, **kwargs),
    'u3fasternetl': lambda *args, **kwargs: load_model_u3('fasternetl', *args, **kwargs),

    # WARNING: different normalization
    #'replxl': lambda *args, **kwargs: load_model_repl('xl', *args, **kwargs),
    # WARNING: different normalization
    # https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html#torchvision.models.efficientnet_v2_s
    #'effl': lambda *args, **kwargs: load_model_eff('l', *args, **kwargs),
}

