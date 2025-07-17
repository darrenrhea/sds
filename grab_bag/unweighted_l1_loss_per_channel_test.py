from unweighted_l1_loss_per_channel import (
     unweighted_l1_loss_per_channel
)
import torch
from colorama import Fore, Style


def test_unweighted_l1_loss_per_channel():
    # If a prediction channel and its corresponding target channel differ by 1 at every pixel, the l1 loss for that channel should be 1.0.
    # If a prediction channel and its corresponding target channel differ by 0.5 at every pixel, the l1 loss for that channel should be 0.5.
        
    a = torch.tensor(
        [
            [ # batch entry 0
                [ # channel 0
                    [0.0, 1.0],
                    [2.0, 3.0],
                ],
                [ # channel 1
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
            ],
            [ # batch entry 1
                [ # channel 0
                    [3.0, 1.0],
                    [4.0, 1.0],
                ],
                [ # channel 1
                    [1.0, 1.0],
                    [0.0, 0.0],
                ],
            ],
        ],
        dtype=torch.float32
    )

    b = torch.tensor(
        [
            [ # batch entry 0
                [ # channel 0
                    [2.0, 0.0],
                    [3.0, 4.0],
                ],
                [ # channel 1
                    [0.5, 0.5],
                    [0.5, 0.5],
                ],
            ],
            [ # batch 1
                [ # channel 0
                    [2.0, 2.0],
                    [3.0, 1.0],
                ],
                [ # channel 1
                    [0.5, 0.5],
                    [0.5, 0.5],
                ],
            ],
        ],
        dtype=torch.float32
    )

    loss_per_target = unweighted_l1_loss_per_channel(
        input=a,
        target=b,
    )
    
    print(f"{Fore.YELLOW}{loss_per_target=}{Style.RESET_ALL}")

    should_be = torch.tensor([1.0000, 0.5000], dtype=torch.float32)
    assert torch.all(loss_per_target == should_be)
    

if __name__ == "__main__":
    test_unweighted_l1_loss_per_channel()
    print(f"{Fore.GREEN}unweighted_l1_loss_per_channel passed tests.{Style.RESET_ALL}")


