import numpy as np
import os
import torch

from src.mlp import MLP
from src.trainer import Trainer
from src.data import AddDataset, MultiplyDataset
from src.experiments import params_add_dataset, params_multiply_dataset


def test_model_init():
    np.random.seed(0)
    torch.manual_seed(0)
    model_args = {
        "number_of_hidden_layers": 1,
        "input_size": 2,
        "hidden_size": 10,
        "activation": torch.nn.Sigmoid(),
    }

    model = MLP(**model_args)
    model.net[0].weight = torch.nn.Parameter(100 * torch.ones(10, 1))
    before = model.net[0].weight.detach().numpy().copy()

    model.initialize()
    after = model.net[0].weight.detach().numpy()

    # However you initialize, it should be with small
    #   numbers but not all zeros
    assert np.std(after) > 0
    assert np.abs(np.mean(after)) < 2


def test_add_dataset():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    dataset = AddDataset(num_examples=1000)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=False)

    # Choose your model's parameters in
    # src.experiment.py:params_add_dataset()
    #   to train a model that can do addition
    model_args, trainer_args = params_add_dataset()
    msg = "Must use torch.nn.MSELoss()"
    assert "loss_func" not in trainer_args, msg

    model = MLP(**model_args)
    model.initialize()

    trainer = Trainer(
        model=model,
        loss_func=torch.nn.MSELoss(),
        **trainer_args)

    _ = trainer.train(data_loader, 200)
    losses = trainer.eval(data_loader)
    msg = "Should learn AddDataset in 200 epochs"
    assert losses[0] < 0.1, msg


def test_saved_add_dataset():

    # Save your model here
    MODEL_FN = "models/test_saved_add_dataset.pt"
    msg = f"Save your model to {MODEL_FN}"
    assert os.path.exists(MODEL_FN), msg
    msg = f"Delete {MODEL_FN} and then save your model there."
    assert os.path.getsize(MODEL_FN) > 0, msg

    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    dataset = AddDataset(num_examples=1000)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=False)

    # Choose your model's parameters in
    # src.experiment.py:params_add_dataset()
    #   to train a model that can do addition
    # We need to use the same model architecture
    #   to load your saved model
    model_args, trainer_args = params_add_dataset()
    msg = "Must use torch.nn.MSELoss()"
    assert "loss_func" not in trainer_args, msg

    model = MLP(**model_args)
    model.load_model(MODEL_FN)

    trainer = Trainer(
        model=model,
        optimizer=print,
        loss_func=torch.nn.MSELoss(),
    )

    losses = trainer.eval(data_loader)
    msg = "Saved model should solve AddDataset"
    assert losses[0] < 0.1, msg


def test_saved_multiply_dataset():

    # Save your model here
    MODEL_FN = "models/test_saved_multiply_dataset.pt"
    msg = f"Save your model to {MODEL_FN}"
    assert os.path.exists(MODEL_FN), msg
    msg = f"Delete {MODEL_FN} and then save your model there."
    assert os.path.getsize(MODEL_FN) > 0, msg

    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    dataset = MultiplyDataset(num_examples=100)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=False)

    # Choose your model's parameters in
    # src.experiment.py:params_multiply_dataset()
    #   to train a model that can do multiplication
    # We need to use the same model architecture
    #   to load your saved model
    model_args, trainer_args = params_multiply_dataset()
    msg = "Must use torch.nn.MSELoss()"
    assert "loss_func" not in trainer_args, msg

    model = MLP(**model_args)
    model.load_model(MODEL_FN)

    trainer = Trainer(
        model=model,
        optimizer=print,
        loss_func=torch.nn.MSELoss(),
    )

    losses = trainer.eval(data_loader)
    msg = "Saved model is expected to do poorly but not as bad as initially"
    assert losses[0] < 4.1e8, msg
