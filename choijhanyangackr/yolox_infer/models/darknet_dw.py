from torch import nn

from .blocks import BaseConv, CSPLayerCustom, FocusCustom, SPPBottleneck


class CSPDarknetDepthwise(nn.Module):
    def __init__(self,
                 depth_multiplier: float,
                 width_multiplier: float,
                 out_features=("dark3", "dark4", "dark5"),
                 act="hard_swish"
                 ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features

        base_channels = int(width_multiplier * 64)  # 64 for L
        base_depth = max(round(depth_multiplier * 3), 1)  # 3 for L

        # stem
        self.stem = FocusCustom(3, base_channels, kernel_size=3, act=act)  # 640 -> 320

        # dark2
        self.dark2 = nn.Sequential(
            BaseConv(base_channels, base_channels * 2, 4, 2, act=act),  # 320 -> 160
            CSPLayerCustom(
                base_channels * 2,  # 128
                base_channels * 2,
                n=base_depth,  # 3
                kernel_size=3,
                dilation=1,
                depthwise=False,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            BaseConv(base_channels * 2, base_channels * 4, 4, 2, act=act),  # 160 -> 80
            CSPLayerCustom(
                base_channels * 4,  # 256
                base_channels * 4,
                n=base_depth * 3,  # 9
                kernel_size=5,
                dilation=1,
                depthwise=True,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            BaseConv(base_channels * 4, base_channels * 8, 4, 2, act=act),  # 80 -> 40
            CSPLayerCustom(
                base_channels * 8,  # 512
                base_channels * 8,
                n=base_depth * 3,  # 9
                kernel_size=5,
                dilation=1,
                depthwise=True,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            BaseConv(base_channels * 8, base_channels * 12, 4, 2, act=act),  # 40 -> 20
            SPPBottleneck(base_channels * 12, base_channels * 12, activation=act),  # 20 -> 20
            CSPLayerCustom(
                base_channels * 12,  # 768
                base_channels * 12,
                n=base_depth,  # 3
                shortcut=False,
                kernel_size=5,
                dilation=1,
                depthwise=True,
                act=act,
            ),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        # outputs["stem"] = x
        x = self.dark2(x)
        # outputs["dark2"] = x

        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}
