# NVIDIA eGPU on Linux

Notes on how to use NVIDIA eGPUs on Linux.

```bash
conda env create -f conda.yaml
conda activate gpu
```



```python
import torch

# How many GPUs?
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

# Name of each GPU
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

# Memory usage
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Memory allocated:", torch.cuda.memory_allocated(device) / 1024**2, "MB")
print("Max memory allocated:", torch.cuda.max_memory_allocated(device) / 1024**2, "MB")
print("Memory reserved:", torch.cuda.memory_reserved(device) / 1024**2, "MB")

# Clear cache
torch.cuda.empty_cache()

# Dosplay CLI tool output nvidia-smi
!nvidia-smi

# For live tracking in Terminal, use built-in loop option
# nvidia-smi -l 1
```

