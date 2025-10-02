# NVIDIA eGPU on Linux

Notes on how to use NVIDIA eGPUs on Linux.

```bash
conda env create -f conda.yaml
conda activate gpu
```



```python
import torch

print(torch.cuda.is_available())        # should be True
print(torch.cuda.device_count())        # should be 2
print(torch.cuda.get_device_name(0))    # T500
print(torch.cuda.get_device_name(1))    # RTX 3060

# Force training on RTX 3060
device = torch.device("cuda:1")
model.to(device)

```

