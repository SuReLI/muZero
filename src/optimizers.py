import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional


class Optimizers:
    """
    """

    def __init__(self, 
                 opt_h: Optional[optim.Optimizer]=None, 
                 opt_g: Optional[optim.Optimizer]=None, 
                 opt_f: Optional[optim.Optimizer]=None) -> None:
        
        self.opt_h = opt_h if opt_h is not None else optim.Adam(self.h.parameters(), lr=self.lr)
        self.opt_g = opt_g if opt_g is not None else optim.Adam(self.g.parameters(), lr=self.lr)
        self.opt_f = opt_f if opt_f is not None else optim.Adam(self.f.parameters(), lr=self.lr)
