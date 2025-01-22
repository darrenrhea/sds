import torch
import torch.nn as nn

class UNet2(nn.Module):
    def __init__(self, in_channels, out_channels, nn_input_height, nn_input_width):
        super().__init__()
        self.nn_input_height = nn_input_height
        self.nn_input_width = nn_input_width

        self.conv1 = self.contract_block(in_channels=in_channels,
                                         out_channels=32,
                                         kernel_size=7,
                                         padding=3)
        self.conv2 = self.contract_block(in_channels=32,
                                         out_channels=32,
                                         kernel_size=3,
                                         padding=1)
        self.conv3 = self.contract_block(in_channels=32,
                                         out_channels=32,
                                         kernel_size=3,
                                         padding=1)

        self.upconv3 = self.expand_block(in_channels=32,
                                         out_channels=32,
                                         kernel_size=3,
                                         padding=1)
        self.upconv2 = self.expand_block(in_channels=32 + 32,
                                         out_channels=16,
                                         kernel_size=3,
                                         padding=1)
        self.upconv1 = self.expand_block(in_channels=16 + 32,
                                         out_channels=out_channels,
                                         kernel_size=3,
                                         padding=1)

    def forward(self, x):  # used to say __call__
        bs = x.shape[0]
        nn_input_height = self.nn_input_height
        nn_input_width = self.nn_input_width

        assert x.shape == (bs, 3, nn_input_height, nn_input_width)
        # downsampling part
        conv1 = self.conv1(x)

        assert conv1.shape == (bs, 32, nn_input_height // 2, nn_input_width // 2)  # 112 x 112

        conv2 = self.conv2(conv1)
        assert conv2.shape == (bs, 32, nn_input_height // 4, nn_input_width // 4)  # 56 x 56

        conv3 = self.conv3(conv2)
        assert conv3.shape == (bs, 32, nn_input_height // 8, nn_input_width // 8)  # 28 x 28

        upconv3 = self.upconv3(conv3)
        assert upconv3.shape == (bs, 32, nn_input_height // 4, nn_input_width // 4)  # 56 x 56

        upconv2 = self.upconv2(torch.cat(tensors=[upconv3, conv2], dim=1))
        assert upconv2.shape == (bs, 16, nn_input_height // 2, nn_input_width // 2)  # 112 x 112

        upconv1 = self.upconv1(torch.cat(tensors=[upconv2, conv1], dim=1))
        assert upconv1.shape == (bs, 2, nn_input_height, nn_input_width)  # 224 x 224

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding),
            torch.nn.BatchNorm2d(out_channels), torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding),
            torch.nn.BatchNorm2d(out_channels), torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(
            torch.nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size,
                            stride=1,
                            padding=padding),
            torch.nn.BatchNorm2d(out_channels), torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels,
                            out_channels,
                            kernel_size,
                            stride=1,
                            padding=padding),
            torch.nn.BatchNorm2d(out_channels), torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     output_padding=1))
        return expand
