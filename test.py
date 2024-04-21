import torch
from safetensors import safe_open

with safe_open("Qwen1.5-0.5B-Chat-GPTQ-Int4/model.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys():
        if 'embed_tokens' in key:
            tensor = f.get_tensor(key)

print(tensor.shape,tensor.dtype,tensor.min(),tensor.max())

tensor = tensor.to(torch.float32)
_min = tensor.min(dim=1)[0]
_max = tensor.max(dim=1)[0]
qscale = torch.max(torch.abs(_min),torch.abs(_max))
qtensor = torch.clip(127 * tensor / qscale.unsqueeze(1),-128,127).to(torch.int8)
qscale = qscale.to(torch.float16)

print(qtensor.shape,qtensor.dtype)
print(qscale.shape,qscale.dtype)

