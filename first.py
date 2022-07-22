import torch
import numpy as np
import torch.nn as nn

class A(nn.Module):
    def __init__(self,nnd):
        self.nnd = 1


# a = np.array([1,2,3])
# print(a)
# b = torch.tensor(a)
# print('b',b)
a = torch.nn.CTCLoss()
print(a)
print(torch.nn.Linear(200,1))