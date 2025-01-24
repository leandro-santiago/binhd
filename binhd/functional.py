import torch
from torch import Tensor

def multibundle(input1: Tensor, input2: Tensor) -> Tensor:
    n = input1.shape[-2] + input2.shape[-2]    
    dtype = input1.dtype
    input1 = input1.sum(dim=-2, dtype=torch.long)
    input2 = input2.sum(dim=-2, dtype=torch.long)

    sample_hv = input1 + input2        
    
    if (n & 1) == 0: # n is even
        bias = torch.empty_like(sample_hv)
        bias.bernoulli_(0.5)
        sample_hv += bias
        n += 1

    threshold = n // 2
    
    return torch.greater(sample_hv, threshold).to(dtype) 
