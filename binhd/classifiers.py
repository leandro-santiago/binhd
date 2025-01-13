import torch
import torch.nn as nn
from torch import Tensor
from tqdm import trange

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
               
        