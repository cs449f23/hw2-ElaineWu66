import torch
import numpy as np


class AddDataset(torch.utils.data.Dataset):
    def __init__(self, num_examples):
        """
        Create a dataset of the form x_1 + x_2 = y

        Save the dataset to class variables.
        You should use torch tensors of dtype float32.
        """
        self.num_examples = num_examples
        data = np.random.randint(-1000, 1000, size=[num_examples, 2])
        label = data.sum(axis=1, keepdims=True)
        # TODO Convert to torch tensors and save these as class variables
        #      so we can load them with self.__getitem__
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.float32)
        # raise NotImplementedError

    def __len__(self):
        return self.num_examples

    def __getitem__(self, item_index):
        """
        Allow us to select items with `dataset[0]`
        Use the class variables you created in __init__.

        Returns (x, y)
            x: the data tensor
            y: the label tensor
        """
        return self.data[item_index],self.label[item_index]
        raise NotImplementedError


class MultiplyDataset(torch.utils.data.Dataset):
    def __init__(self, num_examples):
        """
        Create a dataset of the form x_1 * x_2 = y

        Save the dataset to class variables.
        You should use torch tensors of dtype float32.
        """
        self.num_examples = num_examples
        data = np.random.randint(1, 1000, size=[num_examples, 2])
        label = data.prod(axis=1, keepdims=True)

        # TODO Convert to torch tensors and save these as class variables
        #      so we can load them with self.__getitem__
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.float32)
        # raise NotImplementedError

    def __len__(self):
        return self.num_examples

    def __getitem__(self, item_index):
        """
        Allow us to select items with `dataset[0]`
        Returns (x, y)
            x: the data tensor
            y: the label tensor
        """
        return self.data[item_index],self.label[item_index]
        # raise NotImplementedError
