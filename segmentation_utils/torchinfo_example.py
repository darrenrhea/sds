import torchvision
import torchinfo
from colorama import Fore, Style
model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
    num_classes=2,
    pretrained=False
)


print(f"{Fore.YELLOW}Simply printing the model gives:{Style.RESET_ALL}")
print(model)

print(f"{Fore.YELLOW}\n\n\n\nWhereas torchinfo.summary gives:{Style.RESET_ALL}")

torchinfo.summary(model, (27, 3, 1080, 1920))
print("bye")