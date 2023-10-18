import numpy as np
import os
import torch

from src.data import AddDataset, MultiplyDataset


def test_datasets():
    n = 1000
    msg = "Your dataset should provide tensors of type float32"
    for dataset_cls in [AddDataset, MultiplyDataset]:
        # Iterate through manually
        dataset = dataset_cls(num_examples=n)
        count = 0
        for i in range(len(dataset)):
            x, y = dataset[i]
            assert isinstance(x, torch.Tensor), msg
            assert x.dtype == torch.float32, msg
            assert isinstance(y, torch.Tensor), msg
            count += 1
        assert count == n

        # Iterate through using a DataLoader
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=10, shuffle=False)
        count = 0
        for batch in data_loader:
            x, y = batch
            assert isinstance(x, torch.Tensor), msg
            assert x.dtype == torch.float32, msg
            assert isinstance(y, torch.Tensor), msg
            count += 1
        assert count == n // 10
