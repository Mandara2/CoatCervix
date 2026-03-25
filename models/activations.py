import torch
import torch.nn as nn

class TanhExp(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.exp(x))
    
def get_activation(name):
    name = name.lower()
    if name == "tanhexp":
        return TanhExp()
    raise ValueError(f"Activation not supported: {name}")