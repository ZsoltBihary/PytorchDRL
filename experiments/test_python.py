import torch
from line_profiler_pycharm import profile
# from torchinfo import summary


@profile
def test_func():
    x = torch.randn(5, 5)
    print(x)
    x_CUDA = x.to(device='cuda')
    print(x_CUDA)
    return x_CUDA


print("Hello, StratPopulation!")
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))
    print("Current device index:", torch.cuda.current_device())
else:
    print("No GPU detected. Running on CPU.")

xx = test_func()
