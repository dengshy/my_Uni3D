import torch

def check_pytorch_and_cuda():
    print("=== PyTorch and CUDA Environment Check ===")

    # 检查 PyTorch 版本
    print(f"PyTorch Version: {torch.__version__}")

    # 检查 CUDA 是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        # 检查 CUDA 版本
        cuda_version = torch.version.cuda
        print(f"CUDA Version: {cuda_version}")

        # 检查可用的 GPU 数量
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")

        # 输出每个 GPU 的名称
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")

        # 检查当前使用的 GPU
        current_device = torch.cuda.current_device()
        print(f"Current GPU: {current_device} ({torch.cuda.get_device_name(current_device)})")
    else:
        print("CUDA is not available. Please ensure that CUDA is properly installed and your hardware supports it.")

    print("=== Environment Check Complete ===")

if __name__ == "__main__":
    from torch import nn
    print(nn.Module)
    check_pytorch_and_cuda()
