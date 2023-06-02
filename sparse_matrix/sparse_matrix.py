from typing import List
import torch
import torch.nn as nn
from torch.optim import Adam, SparseAdam, SGD
import numpy as np
import random
from timeit import timeit


class SparseLinear(nn.Module):
    '''
    Implements a Sparse Linear Layer, the matrix multiplication uses a sparse matrix
    num_inputs, num_outputs: This creates a sparse matrix with num_inputs rows and num_outputs columns
    row_idxs: row indices between in range [0, num_inputs]
    col_idxs: column indices between in range [0, num_outputs]
    each non empty element of the matrix must be represented as row_index and column_index
    example 
    [[7, 0, 3], 
     [0, 6, 0]]
    row_indices = [0, 0, 1]
    col_indices = [0, 2, 1]
    which means the elements are [0, 0], [0, 2], [1, 1]
    '''
    def __init__(self, num_inputs, num_outputs, row_idxs: torch.Tensor, col_idxs: torch.Tensor):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.index_matrix = torch.stack((row_idxs, col_idxs))
        weights, bias = self.initializeWeights()
        self.weights = nn.Parameter(weights)
        self.bias = nn.Parameter(bias)

    def initializeWeights(self):
        weights = torch.rand(self.index_matrix.shape[1], dtype=torch.float32)
        bias = torch.rand((self.num_outputs,1), dtype=torch.float32)
        return weights, bias

    def to(self, device):
        self.device = device
        super().to(device)
        self.index_matrix = self.index_matrix.to(device)

    def forward(self, x):
        A = torch.sparse_coo_tensor(self.index_matrix, self.weights, (self.num_inputs, self.num_outputs), dtype=torch.float32)
        y = torch.sparse.mm(A.T, x.T) + self.bias
        return y.T

def train(model, x, y_true, num_epochs: int):
    LR = 0.02
    model.train()
    params = [p for p in model.parameters()]
    optimizer = Adam(params, LR)
    for epoch in range(0, num_epochs):
        y_pred = model(x)
        loss = torch.mean((y_true - y_pred) ** 2)
        print(f'epoch={epoch}, loss={loss.detach()}')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def example_02():
    '''
    compare performance, using either matmul or torch.sparse.mm
    use a large matrix 60000 x 4000
    time comparison on CPU / GPU: RTX 2080 TI 
    | torch.Linear   | cpu     | 96 sec    |
    | SparseLinear   | cpu     | 53 sec    |
    | torch.Linear   | gpu     |  3.10 sec |
    | SparseLinear   | gpu     |  1.17 sec | 
    '''
    use = 'cpu'
    #use = 'cuda'
    device = torch.device(use)
    num_inputs = 60000
    num_outputs = 4000
    count = 50000
    # num_inputs = 10
    # num_outputs = 6
    # count = 6
    row_idxs = torch.randint(0, num_inputs, (count,))
    col_idxs = torch.randint(0, num_outputs, (count,))
    model = SparseLinear(num_inputs, num_outputs, row_idxs, col_idxs)
    #model = nn.Linear(num_inputs, num_outputs, bias=True)
    model.to(device)
    # x must have shape(num_samples, num_inputs)
    # y must have shape(num_samples, num_outputs)
    num_samples = 1000
    x = torch.rand((num_samples, num_inputs)).to(device)
    y_true = torch.rand((num_samples, num_outputs)).to(device)
    print(timeit(lambda: train(model, x, y_true, 20), number=1))
    pass

def main():
    example_02()

if __name__ == '__main__':
    main()


