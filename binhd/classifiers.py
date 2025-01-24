import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from tqdm import trange
from torchhd.classifiers import Centroid
from torchhd.embeddings import Sinusoid

class BinHD(nn.Module):
    def __init__(
        self,
        n_dimensions: int,
        n_classes: int,
        *,
        epochs: int = 30,
        device: torch.device = None,        
    ) -> None:
        super().__init__()

        self.n_dimensions = n_dimensions
        self.n_classes = n_classes
        self.epochs = epochs
        self.classes_counter = torch.empty((n_classes, n_dimensions), device=device, dtype=torch.int8)
        self.classes_hv = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.classes_counter)        
        
    def fit(self, input: Tensor, target: Tensor):
        input = 2 * input - 1
        self.classes_counter.index_add_(0, target, input)
        #self.classes_hv = torch.where(self.classes_counter >= 0, 1, 0)
        self.classes_hv = self.classes_counter.clamp(min=0, max=1)

    def fit_adapt(self, input: Tensor, target: Tensor):
        for _ in trange(0, self.epochs, desc="fit"):            
            self.adapt(input, target)

    def adapt(self, input: Tensor, target: Tensor):
        pred = self.predict(input)
        is_wrong = target != pred
        
        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            return

        input = input[is_wrong]
        input = 2 * input - 1
        target = target[is_wrong]
        pred = pred[is_wrong]
        
        self.classes_counter.index_add_(0, target, input, alpha=1)        
        self.classes_counter.index_add_(0, pred, input, alpha=-1)        
        self.classes_hv = torch.where(self.classes_counter >= 0, 1, 0)        
    
    def forward(self, samples: Tensor) -> Tensor:
        response = torch.empty((self.n_classes, samples.shape[0]), dtype=torch.int8)
        
        for i in range(self.n_classes):
            # Hamming Distance = SUM(XOR(a, b))
            response[i] = torch.sum(torch.bitwise_xor(samples, self.classes_hv[i]), dim=1)  # Hamming distance          
        
        return response.transpose_(0,1)

    def predict(self, samples: Tensor) -> Tensor:
        return torch.argmin(self(samples), dim=-1)
               

class NeuralHD(nn.Module):
    r"""Implements `Scalable edge-based hyperdimensional learning system with brain-like neural adaptation <https://dl.acm.org/doi/abs/10.1145/3458817.3480958>`_.

    Args:
        n_features (int): Size of each input sample.
        n_dimensions (int): The number of hidden dimensions to use.
        n_classes (int): The number of classes.
        regen_freq (int, optional): The frequency in epochs at which to regenerate hidden dimensions.
        regen_rate (int, optional): The fraction of hidden dimensions to regenerate.
        epochs (int, optional): The number of iteration over the training data.
        lr (float, optional): The learning rate.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.

    """

    model: Centroid
    encoder: Sinusoid

    def __init__(
        self,
        n_features: int,
        n_dimensions: int,
        n_classes: int,
        *,
        regen_freq: int = 20,
        regen_rate: float = 0.04,
        epochs: int = 120,
        lr: float = 0.37,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.n_dimensions = n_dimensions
        self.n_classes = n_classes
        self.regen_freq = regen_freq
        self.regen_rate = regen_rate
        self.epochs = epochs
        self.lr = lr

        self.encoder = Sinusoid(n_features, n_dimensions, device=device, dtype=dtype)
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)

    def fit(self, input: Tensor, target: Tensor):
        encoded = self.encoder(input)
        n_regen_dims = math.ceil(self.regen_rate * self.n_dimensions)
        self.model.add(encoded, target)

        for epoch_idx in trange(1, self.epochs, desc="fit"):
            encoded = self.encoder(input)
            self.model.add_adapt(encoded, target, lr=self.lr)

            # Regenerate feature dimensions
            if (epoch_idx % self.regen_freq) == (self.regen_freq - 1):
                weight = F.normalize(self.model.weight, dim=1)
                scores = torch.var(weight, dim=0)

                regen_dims = torch.topk(scores, n_regen_dims, largest=False).indices
                self.model.weight.data[:, regen_dims].zero_()
                self.encoder.weight.data[regen_dims, :].normal_()
                self.encoder.bias.data[:, regen_dims].uniform_(0, 2 * math.pi)

        return self

    def forward(self, samples: Tensor) -> Tensor:
        return self.model(self.encoder(samples))

    def predict(self, samples: Tensor) -> Tensor:
        return torch.argmax(self(samples), dim=-1)