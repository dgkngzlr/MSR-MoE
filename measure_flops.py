import argparse

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count

from models.oursnet.model import OursNet


def main():
    parser = argparse.ArgumentParser(description="Measure GFLOPs of OursNet")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--rgb_ch", type=int, default=3)
    parser.add_argument("--ir_ch", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=9)
    parser.add_argument(
        "--model_name",
        type=str,
        default="convnextv2_tiny.fcmae_ft_in22k_in1k",
    )
    args = parser.parse_args()

    model = OursNet(
        rgb_ch=args.rgb_ch,
        ir_ch=args.ir_ch,
        num_classes=args.num_classes,
        freeze_layer=0,
        model_name=args.model_name,
    )
    model.eval()

    rgb = torch.zeros(1, args.rgb_ch, args.height, args.width)
    ir = torch.zeros(1, args.ir_ch, args.height, args.width)

    flops = FlopCountAnalysis(model, (rgb, ir))
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)

    params = parameter_count(model)

    print(flop_count_table(flops, max_depth=3))
    print(f"\nResolution : {args.height}x{args.width}")
    print(f"GFLOPs     : {flops.total() / 1e9:.2f}")
    print(f"Params (M) : {params[''] / 1e6:.2f}")


if __name__ == "__main__":
    main()
