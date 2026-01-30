import torch
import time

# 1. 基本環境檢查
print("--- 環境檢查 ---")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # 2. 顯卡資訊
    device = torch.device("cuda")
    print(f"當前顯卡: {torch.cuda.get_device_name(0)}")
    
    # 3. 效能測試：大規模矩陣相乘 (Matrix Multiplication)
    print("\n--- 效能測試 (GPU) ---")
    size = 10000  # 建立 10000x10000 的大矩陣
    
    # 在 GPU 上建立隨機矩陣
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # 預熱 (Warm up)
    torch.matmul(a, b)
    
    # 開始計時
    start_time = time.time()
    
    # 執行運算
    result = torch.matmul(a, b)
    
    # 同步 CUDA (確保運算完成才停止計時)
    torch.cuda.synchronize()
    
    end_time = time.time()
    
    print(f"10000x10000 矩陣相乘耗時: {end_time - start_time:.4f} 秒")
    print("測試完成，RTX 5070 運作正常！")
else:
    print("錯誤：無法偵測到 CUDA，請檢查驅動程式與 PyTorch 版本。")