# check_gpu.py
import torch

print("--- Checking GPU Availability ---")
is_available = torch.cuda.is_available()
print(f"torch.cuda.is_available(): {is_available}")

if is_available:
    print(f"\n✅  SUCCESS: CUDA IS AVAILABLE!  ✅")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print(f"\n❌  ERROR: CUDA IS NOT AVAILABLE  ❌")
    print("This confirms that PyTorch cannot see your GPU.")
    print("This is why 'config.DEVICE' is set to 'cpu'.")
    print("\n--- How to Fix ---")
    print("The most likely fix is to re-install PyTorch with the correct CUDA version.")
    print("1. Go to the official PyTorch website: https://pytorch.org/get-started/locally/")
    print("2. Select the correct options for your system (Stable, Windows, Pip, Python, CUDA 11.8 or 12.1).")
    print("3. Copy the generated command and run it in your terminal.")
    print("   (It will look something like: pip3 install torch torchvision torchaudio --index-url ...)")