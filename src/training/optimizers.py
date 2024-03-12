import torch.nn as nn
import torch.optim as optim
from typing import Optional


class Optimizers:
    """
    Class in which the optimizers for the three models are stored.
    By default, the optimizer is Adam with learning rate 0.001.
    """

    def __init__(self,
                    h: nn.Module,
                    g: nn.Module,
                    f: nn.Module,
                    opt_h: Optional[optim.Optimizer]=None, 
                    opt_g: Optional[optim.Optimizer]=None, 
                    opt_f: Optional[optim.Optimizer]=None,   
                    lr: Optional[float]=0.001) -> None:
        
        self.lr = lr
        self.opt_h = opt_h if opt_h is not None else optim.Adam(h.parameters(), lr=self.lr)
        self.opt_g = opt_g if opt_g is not None else optim.Adam(g.parameters(), lr=self.lr)
        self.opt_f = opt_f if opt_f is not None else optim.Adam(f.parameters(), lr=self.lr)
        print(f"opt types: {type(self.opt_h)}, {type(self.opt_g)}, {type(self.opt_f)}")
    @classmethod
    def basic_optimizer(cls, h: nn.Module, g: nn.Module, f: nn.Module) -> 'Optimizers':
        lr = 0.001
        return cls(h, g, f, optim.Adam(h.parameters(), lr=lr), optim.Adam(g.parameters(), lr=lr), optim.Adam(f.parameters(), lr=lr))