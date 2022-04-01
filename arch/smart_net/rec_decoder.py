import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import modules as md

class UpsampleBlock(nn.Module):
    def __init__(self, scale, input_channels, output_channels, ksize=1):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(input_channels, output_channels*(scale**2), kernel_size=1, stride=1, padding=ksize//2),
            nn.PixelShuffle(upscale_factor=scale)
        )

    def forward(self, input):
        return self.upsample(input)

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=False,
            attention_type=None,
    ):
        super().__init__()

        self.upsample   = UpsampleBlock(scale=2, input_channels=in_channels, output_channels=in_channels)

        self.attention1 = md.Attention(attention_type, in_channels=in_channels)
        self.conv1      = md.Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2      = md.Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)
        

    def forward(self, x):
        x = self.upsample(x)

        x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class Last_DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x):
        x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)





class AE_Decoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            use_batchnorm=False,
            attention_type=None,
            center=False,
    ):
        super().__init__()
        encoder_channels = encoder_channels[1:]    # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels   = [head_channels] + list(decoder_channels[:-1])
        out_channels  = decoder_channels    # (256, 128, 64, 32)

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [ DecoderBlock(in_ch, out_ch, **kwargs) for in_ch, out_ch in zip(in_channels, out_channels) ]
        self.blocks = nn.ModuleList(blocks)
        self.last_block = Last_DecoderBlock(in_channels=32, out_channels=16, use_batchnorm=True, attention_type='scse')



    def forward(self, features):
        x = self.center(features)

        for decoder_block in self.blocks:
            x = decoder_block(x)

        x = self.last_block(x)

        return x
