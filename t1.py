import time
import torch
import numpy as np

x = torch.tensor(0.01)
a = torch.log(x / (1 - x))
print(a)
b = torch.logit(torch.tensor(0.01))
print(b)