import torch
import torch.nn.functional as F
from colorama import Fore, Style
from weighted_mse_loss import weighted_mse_loss

def test_weighted_mse_loss():
    a = torch.tensor(
        [
            [ # batch entry 0
                [ # channel 0
                    [0.0, 1.0],
                    [2.0, 3.0],
                ]
            ],
            [ # batch entry 1
                [ # channel 0
                    [3.0, 1.0],
                    [4.0, 1.0],
                ]
            ],
        ],
        dtype=torch.float32
    )

    b = torch.tensor(
        [
            [ # batch entry 0
                [ # channel 0
                    [2.0, 3.0],
                    [5.0, 7.0],
                ]
            ],
            [ # batch 1
                [ # channel 0
                    [2.0, 7.0],
                    [1.0, 8.0],
                ]
            ],
        ],
        dtype=torch.float32
    )

    importance_weights = torch.tensor.fu(
        [
            [ # batch entry 0
                [ # channel 0
                    [0.5, 0.5],
                    [0.5, 0.5],
                ]
            ],
            [ # batch 1
                [ # channel 0
                    [0.5, 0.5],
                    [0.5, 0.5],
                ]
            ],
        ],
        dtype=torch.float32
    )

    mse_loss = torch.nn.MSELoss()

    reproduced_loss_value = weighted_mse_loss(
        input=a,
        target=b,
        importance_weights=importance_weights
    )

    functional_loss_value = F.mse_loss(a, b, reduction='mean')
    
    module_loss_value = mse_loss(a, b)

    print(f"{Fore.YELLOW}{reproduced_loss_value=}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{functional_loss_value=}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{module_loss_value=}{Style.RESET_ALL}")


    assert torch.abs(reproduced_loss_value - functional_loss_value) < 0.0000001
    assert torch.abs(reproduced_loss_value - module_loss_value) < 0.0000001

  

