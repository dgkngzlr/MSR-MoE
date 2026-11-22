import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model


class ConvNeXtV2Backbone(nn.Module):
    def __init__(self, in_ch, freeze_layer, model_name="convnextv2_tiny.fcmae_ft_in22k_in1k"):
        super().__init__()
        self.in_ch = in_ch
        self.freeze_layer = freeze_layer
        self.model_name = model_name

        if freeze_layer < 0 or freeze_layer > 4:
            raise ValueError(f"freeze_layer must be in [0, 4], got {freeze_layer}")

        self.model = create_model(
            model_name,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        self.channels = self.model.feature_info.channels()

        if in_ch == 1:
            self._adapt_to_grayscale()

        if freeze_layer != 0:
            self._freeze_layers()

    def _get_layer_groups(self):
        """
        Returns layer groups where stem is always bundled with stages_0.
            1 -> stem + stages_0
            2 -> stem + stages_0 + stages_1
            3 -> stem + stages_0 + stages_1 + stages_2
            4 -> stem + stages_0 + stages_1 + stages_2 + stages_3
        """
        return [
            [self.model.stem_0, self.model.stem_1, self.model.stages_0],  # group 1
            [self.model.stages_1],                                         # group 2
            [self.model.stages_2],                                         # group 3
            [self.model.stages_3],                                         # group 4
        ]

    def _adapt_to_grayscale(self):
        if hasattr(self.model, 'stem_0'):
            old_conv = self.model.stem_0
        else:
            raise RuntimeError("Could not find the first Conv2d layer in the model stem.")

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=(old_conv.bias is not None),
            padding_mode=old_conv.padding_mode,
        )

        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        self.model.stem_0 = new_conv

    def _freeze_layers(self):
        for group in self._get_layer_groups()[:self.freeze_layer]:
            for module in group:
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False

    def unfreeze_layers(self):
        for module in self.model.modules():
            module.train()
        for p in self.model.parameters():
            p.requires_grad = True

    def train(self, mode=True):
        """Prevent frozen layers from switching back to train mode."""
        super().train(mode)
        if mode and self.freeze_layer > 0:
            for group in self._get_layer_groups()[:self.freeze_layer]:
                for module in group:
                    module.eval()
        return self

    def forward(self, x):
        feats = self.model(x)
        return {
            "p4": feats[0],
            "p8": feats[1],
            "p16": feats[2],
            "p32": feats[3],
        }

class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        output = torch.sigmoid(output) * x
        return output

class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max.view(b,c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b,c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = torch.sigmoid(output) * x
        return output
    
class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x
    
class MOE2(nn.Module):
    """Mixture of Experts for 2 inputs"""
    def __init__(self, nf=32, dropout_ratio=0.0):
        super(MOE2, self).__init__()
        self.nf = nf  
        self.dropout_ratio = dropout_ratio  
        
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        
        self.fc_a = nn.Conv2d(nf, nf, kernel_size=1, bias=True, stride=1, padding=0, groups=1)
        self.fc_b = nn.Conv2d(nf, nf, kernel_size=1, bias=True, stride=1, padding=0, groups=1)
        self.fuse = nn.Conv2d(nf, nf, kernel_size=1, bias=True, stride=1, padding=0, groups=1)
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, x):
        
        a, b = x[0], x[1]
              
        wa = self.fc_a(a)  
        wb = self.fc_b(b)    
        wm = self.softmax(torch.stack([wa, wb], dim=0))
        
        a = a * wm[0,:].squeeze(0)  
        b = b * wm[1,:].squeeze(0)  
        
        out = a + b  
        
        if self.dropout_ratio > 0:  
            out = self.dropout(out)  
        
        out = self.fuse(out) + out    
        
        return out

class NodeConv(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=4):
        super().__init__()
        # Make reduction ratio adaptive to avoid issues with small channel sizes
        r = max(1, min(reduction_ratio, in_channels // 2))
        self.cbam = CBAM(in_channels, r)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.cbam(x)
        x = self.conv(x)
        return x 

class Mixer(nn.Module):

    def __init__(self, nb_channels):

        super().__init__()

        self.nb_channels = nb_channels

        # first level
        self.conv0_0 = NodeConv(nb_channels[0], nb_channels[0])
        self.conv0_1 = NodeConv(nb_channels[0] + nb_channels[1], nb_channels[0])
        self.conv0_2 = NodeConv(2 * nb_channels[0] + nb_channels[1], nb_channels[0])
        self.conv0_3 = NodeConv(
            3 * nb_channels[0] + nb_channels[1] + nb_channels[2] + nb_channels[3],
            nb_channels[0],
        )

        # second level
        self.conv1_0 = NodeConv(nb_channels[0] + nb_channels[1], nb_channels[1])
        self.conv1_1 = NodeConv(
            nb_channels[0] + nb_channels[1] + nb_channels[2], nb_channels[1]
        )
        self.conv1_2 = NodeConv(
            nb_channels[0] + 2 * nb_channels[1] + nb_channels[2] + nb_channels[3],
            nb_channels[1],
        )

        # third level
        self.conv2_0 = NodeConv(nb_channels[1] + nb_channels[2], nb_channels[2])
        self.conv2_1 = NodeConv(
            nb_channels[1] + nb_channels[2] + nb_channels[3], nb_channels[2]
        )

        # fourth level
        self.conv3_0 = NodeConv(nb_channels[2] + nb_channels[3], nb_channels[3])

    def _upsample(self, x, scale_factor=2):
        return F.interpolate(
            x, scale_factor=scale_factor, mode="bilinear", align_corners=False
        )

    def _downsample(self, x, scale_factor=0.5):
        return F.interpolate(
            x, scale_factor=scale_factor, mode="bilinear", align_corners=False
        )

    def forward(self, x):

        x0, x1, x2, x3 = x

        x0_0 = self.conv0_0(x0)
        x1_0 = self.conv1_0(torch.cat([self._downsample(x0_0), x1], dim=1))
        x2_0 = self.conv2_0(torch.cat([self._downsample(x1_0), x2], dim=1))
        x3_0 = self.conv3_0(torch.cat([self._downsample(x2_0), x3], dim=1))

        x0_1 = self.conv0_1(torch.cat([x0_0, self._upsample(x1_0)], dim=1))
        x1_1 = self.conv1_1(
            torch.cat([self._downsample(x0_1), x1_0, self._upsample(x2_0)], dim=1)
        )
        x2_1 = self.conv2_1(
            torch.cat([self._downsample(x1_1), x2_0, self._upsample(x3_0)], dim=1)
        )

        x0_2 = self.conv0_2(torch.cat([x0_1, x0_0, self._upsample(x1_1)], dim=1))
        x1_2 = self.conv1_2(
            torch.cat(
                [
                    self._downsample(x0_2),
                    x1_1,
                    x1_0,
                    self._upsample(x2_1),
                    self._upsample(x3_0, scale_factor=4),
                ],
                dim=1,
            )
        )

        x0_3 = self.conv0_3(
            torch.cat(
                [
                    x0_2,
                    x0_1,
                    x0_0,
                    self._upsample(x1_2),
                    self._upsample(x2_1, scale_factor=4),
                    self._upsample(x3_0, scale_factor=8),
                ],
                dim=1,
            )
        )

        return x0_3, x1_2, x2_1, x3_0

class Decoder(nn.Module):

    def __init__(
        self, nb_channels, num_classes, reduction_ratio=4
    ):
        super().__init__()

        self.num_classes = num_classes
        self.nb_channels = nb_channels

        self.moe0 = MOE2(nb_channels[0], dropout_ratio=0.0)
        self.moe1 = MOE2(nb_channels[1], dropout_ratio=0.0)
        self.moe2 = MOE2(nb_channels[2], dropout_ratio=0.0)
        self.moe3 = MOE2(nb_channels[3], dropout_ratio=0.0)

        self.layer3_33 = nn.Sequential(
            nn.Conv2d(nb_channels[3], nb_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(nb_channels[3]),
            nn.GELU(),
        )
        self.layer3_11 = nn.Sequential(
            nn.Conv2d(nb_channels[3], self.num_classes, kernel_size=1),
        )

        self.layer2_33 = nn.Sequential(
            nn.Conv2d(
                nb_channels[2] + nb_channels[3],
                nb_channels[2],
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(nb_channels[2]),
            nn.GELU(),
        )
        self.layer2_11 = nn.Sequential(
            nn.Conv2d(nb_channels[2], self.num_classes, kernel_size=1),
        )

        self.layer1_33 = nn.Sequential(
            nn.Conv2d(
                nb_channels[1] + nb_channels[2],
                nb_channels[1],
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(nb_channels[1]),
            nn.GELU(),
        )

        self.layer1_11 = nn.Sequential(
            nn.Conv2d(nb_channels[1], self.num_classes, kernel_size=1),
        )

        self.layer0_33 = nn.Sequential(
            nn.Conv2d(
                nb_channels[0] + nb_channels[1],
                nb_channels[0],
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(nb_channels[0]),
            nn.GELU(),
        )

        self.layer0_11 = nn.Sequential(
            nn.Conv2d(nb_channels[0], self.num_classes, kernel_size=1),
        )

        self.layer_last_33 = nn.Sequential(
            nn.Conv2d(nb_channels[0], nb_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(nb_channels[0]),
            nn.GELU(),
        )

        self.layer_last_11 = nn.Sequential(
            nn.Conv2d(nb_channels[0], self.num_classes, kernel_size=1),
        )

    def forward(self, x_rgb, x_ir):

        x_rgb_0, x_rgb_1, x_rgb_2, x_rgb_3 = x_rgb
        x_ir_0, x_ir_1, x_ir_2, x_ir_3 = x_ir

        x0 = self.moe0([x_rgb_0, x_ir_0])
        x1 = self.moe1([x_rgb_1, x_ir_1])
        x2 = self.moe2([x_rgb_2, x_ir_2])
        x3 = self.moe3([x_rgb_3, x_ir_3])

        x3_33_out = self.layer3_33(x3)
        x3_11_out = self.layer3_11(x3_33_out)

        x2_33_out = self.layer2_33(
            torch.cat([x2, F.interpolate(x3_33_out, scale_factor=2, mode="bilinear", align_corners=False)], dim=1)
        )
        x2_11_out = self.layer2_11(x2_33_out)

        x1_33_out = self.layer1_33(
            torch.cat([x1, F.interpolate(x2_33_out, scale_factor=2, mode="bilinear", align_corners=False)], dim=1)
        )
        x1_11_out = self.layer1_11(x1_33_out)

        x0_33_out = self.layer0_33(
            torch.cat([x0, F.interpolate(x1_33_out, scale_factor=2, mode="bilinear", align_corners=False)], dim=1)
        )
        x0_11_out = self.layer0_11(x0_33_out)

        x_last_33_out = self.layer_last_33(F.interpolate(x0_33_out, scale_factor=4, mode="bilinear", align_corners=False))
        x_last_11_out = self.layer_last_11(x_last_33_out)

        return x_last_11_out

class OursNet(nn.Module):
    def __init__(self, rgb_ch, ir_ch, num_classes, freeze_layer, model_name="convnextv2_tiny.fcmae_ft_in22k_in1k"):
        super().__init__()

        self.rgb_ch = rgb_ch
        self.ir_ch = ir_ch
        self.num_classes = num_classes

        self.rgb_encoder = ConvNeXtV2Backbone(in_ch=rgb_ch, freeze_layer=freeze_layer, model_name=model_name)
        self.ir_encoder = ConvNeXtV2Backbone(in_ch=ir_ch, freeze_layer=freeze_layer, model_name=model_name)
        self.rgb_mixer = Mixer(self.rgb_encoder.channels)
        self.ir_mixer = Mixer(self.ir_encoder.channels)
        self.decoder = Decoder(self.rgb_encoder.channels, num_classes)

    def forward(self, x_rgb, x_ir):

        # Handle IR channel constraint safely
        if self.ir_ch == 1:
            x_ir = x_ir[:, 0:1, :, :]

        x_rgb = self.rgb_encoder(x_rgb)
        x_rgb_0 = x_rgb["p4"]
        x_rgb_1 = x_rgb["p8"]
        x_rgb_2 = x_rgb["p16"]
        x_rgb_3 = x_rgb["p32"]

        x_ir = self.ir_encoder(x_ir)
        x_ir_0 = x_ir["p4"]
        x_ir_1 = x_ir["p8"]
        x_ir_2 = x_ir["p16"]
        x_ir_3 = x_ir["p32"]

        x_rgb_0, x_rgb_1, x_rgb_2, x_rgb_3 = self.rgb_mixer(
            [x_rgb_0, x_rgb_1, x_rgb_2, x_rgb_3]
        )
        x_ir_0, x_ir_1, x_ir_2, x_ir_3 = self.ir_mixer([x_ir_0, x_ir_1, x_ir_2, x_ir_3])

        return self.decoder(
            [x_rgb_0, x_rgb_1, x_rgb_2, x_rgb_3], [x_ir_0, x_ir_1, x_ir_2, x_ir_3]
        )

if __name__ == "__main__":

    rgb_img = torch.randn(1, 3, 512, 512)
    thermal_img = torch.randn(1, 1, 512, 512) # Thermal de 3 kanal varsayıldı

    model = OursNet(rgb_ch=3, ir_ch=1, num_classes=9, freeze_layer=4)
    output = model(rgb_img, thermal_img)
    
    print("Input Size:", rgb_img.shape)
    print("Output Size:", output.shape)