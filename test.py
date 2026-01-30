import torch
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"顯卡名稱: {torch.cuda.get_device_name(0)}")