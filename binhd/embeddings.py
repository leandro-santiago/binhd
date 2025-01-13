import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter

import torchhd.functional as functional
from torchhd.tensors.bsc import BSCTensor


__all__ = [
    "ScatterCode",    
]

def bin_level(
    num_vectors: int,
    dimensions: int,    
    *,
    requires_grad=False,
    **kwargs,
) -> BSCTensor:
    """Creates a set of binary level based on Scatter Code.    

    Args:
        num_vectors (int): the number of hypervectors to generate.
        dimensions (int): the dimensionality of the hypervectors.
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` depends on VSATensor.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.
    
    """
    vsa_tensor = BSCTensor

    num_flipped_bits = dimensions // num_vectors
    idx_mapping = torch.randperm(dimensions)
    
    hv = torch.empty(
        num_vectors,
        dimensions,
        dtype=kwargs["dtype"],
        device=kwargs["device"],
    ).as_subclass(vsa_tensor)

    base = vsa_tensor.random(
        1,
        dimensions,
        **kwargs,
    )

    hv[0] = base[0] # min hyper-vector
    slice_start = 0
    slice_end = num_flipped_bits

    for i in range(1, num_vectors):
        # Mark adding by 2 the num_flipped_bits position from base hyper-vector
        base.scatter_(1, idx_mapping[slice_start:slice_end].unsqueeze(0), 2, reduce='add')
        hv[i] = base[0]
        slice_start = slice_end
        slice_end += num_flipped_bits

    hv = hv.where(hv < 2, torch.logical_not(hv - 2))    
    hv.requires_grad = requires_grad
    return hv


class ScatterCode(nn.Embedding):
    """Embedding wrapper around :func:`bin_level`.

    Class inherits from `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>`_ and supports the same keyword arguments.

    Args:
        num_embeddings (int): the number of hypervectors to generate.
        embedding_dim (int): the dimensionality of the hypervectors.
        vsa: (``VSAOptions``, optional): specifies the hypervector type to be instantiated. Default: ``"MAP"``.
        low (float, optional): The lower bound of the real number range that the levels represent. Default: ``0.0``
        high (float, optional): The upper bound of the real number range that the levels represent. Default: ``1.0``
        dtype (``torch.dtype``, optional): the desired data type of returned tensor. Default: if ``None`` uses default of ``VSATensor``.
        device (``torch.device``, optional):  the desired device of returned tensor. Default: if ``None``, uses the current device for the default tensor type (see torch.set_default_tensor_type()). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Values outside the interval between low and high are clipped to the binary bound.
    """

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "low",
        "high",        
    ]

    low: float
    high: float
    vsa = "BSC"

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        low: float = 0.0,
        high: float = 1.0,
        requires_grad: bool = False,        
        device=None,                
    ) -> None:
        factory_kwargs = {"device": device, "dtype": torch.int8}
        # Have to call Module init explicitly in order not to use the Embedding init
        nn.Module.__init__(self)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.low = low
        self.high = high 
        self.padding_idx = None # required by nn.Embedding 
        self.max_norm = None # required by nn.Embedding   
        self.norm_type = 2 # required by nn.Embedding     
        self.scale_grad_by_freq = False # required by nn.Embedding 
        self.sparse = False # required by nn.Embedding     

        embeddings = bin_level(
            num_embeddings,
            embedding_dim,            
            **factory_kwargs            
        )
        # Have to provide requires grad at the creation of the parameters to
        # prevent errors when instantiating a non-float embedding
        self.weight = Parameter(embeddings, requires_grad=requires_grad)

    def reset_parameters(self) -> None:
        factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}

        with torch.no_grad():
            embeddings = bin_level(
                self.num_embeddings,
                self.embedding_dim,                
                **factory_kwargs,
                **self.vsa_kwargs,
            )
            self.weight.copy_(embeddings)

    def forward(self, input: Tensor) -> Tensor:
        index = functional.value_to_index(
            input, self.low, self.high, self.num_embeddings
        )
        index = index.clamp(min=0, max=self.num_embeddings - 1)
        vsa_tensor = functional.get_vsa_tensor_class(self.vsa)
        return super().forward(index).as_subclass(vsa_tensor)