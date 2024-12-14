import torch

def gpu_tensor_test():
    print("=== GPU Tensor Test ===")

    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot run GPU test.")
        return

    # 在 GPU 上创建一个张量
    device = torch.device("cuda")
    tensor = torch.rand((3, 3), device=device)
    print("Tensor on GPU:")
    print(tensor)

    # 在 GPU 上进行张量计算
    result = tensor @ tensor.T  # 矩阵乘法
    print("Result of tensor multiplication on GPU:")
    print(result)

    print("=== GPU Test Complete ===")

if __name__ == "__main__":
    gpu_tensor_test()
