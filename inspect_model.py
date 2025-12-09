from hear21passt.base import get_basic_model
import torch

print("\n--- get_basic_model(mode='all') ---")
try:
    basic_model = get_basic_model(mode="all")
    print("Basic Model (all) type:", type(basic_model))
    
    x = torch.randn(1, 32000*10)
    y = basic_model(x)
    # y might be a dict or tuple
    if isinstance(y, dict):
        print("Output keys:", y.keys())
        for k, v in y.items():
            if isinstance(v, torch.Tensor):
                print(f"Key {k}: {v.shape}")
    elif isinstance(y, torch.Tensor):
        print("Output shape:", y.shape)
    else:
        print("Output type:", type(y))
        
except Exception as e:
    print("get_basic_model(mode='all') failed:", e)



