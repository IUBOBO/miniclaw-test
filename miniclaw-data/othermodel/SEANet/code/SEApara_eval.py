import torch
from thop import profile
from thop import clever_format

# from model import Tripmodel as Net
from models import SEANet

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SEANet().to(device)
    model.eval()

    # (Batch_size=1, Channels=3, H=512, W=512)
    dummy_input = torch.randn(1, 3, 512, 512).to(device)

    macs, params = profile(model, inputs=(dummy_input,), verbose=False)

    macs_fmt, params_fmt = clever_format([macs, params], "%.2f")

    flops_g = (macs * 2) / 1e9

    print("\n" + "=" * 40)
    print(f"  {model.__class__.__name__}")
    print(f": {dummy_input.shape}")
    print(f" (Params): {params_fmt}")
    print(f" (MACs):   {macs_fmt}")
    print(f" (FLOPs):  {flops_g:.2f} G")
    print("=" * 40)