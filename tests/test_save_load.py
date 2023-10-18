import datetime
import numpy as np
import os
import re
import torch

from src.mlp import MLP
from src.trainer import Trainer
from src.data import AddDataset, MultiplyDataset
from src.experiments import params_add_dataset, params_multiply_dataset

model_args = {
    "number_of_hidden_layers": 1,
    "input_size": 2,
    "hidden_size": 1,
    "activation": torch.nn.Sigmoid(),
}

trainer_args = {
    "optimizer": torch.optim.SGD,
    "loss_func": torch.nn.MSELoss(),
    "lr": 0.1,
}


def test_load_model():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    model = MLP(**model_args)

    # This model is provided with the repository;
    #    you don't need to modify it
    fn = f"models/test_load_model.pt"
    model.load_model(fn)

    # It should load correctly
    first_layer = model.net[0].weight.detach().numpy()
    reference = np.array([0.449, 0.19120623])
    assert np.allclose(first_layer[0], reference)


def test_save_load_model():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    dataset = AddDataset(num_examples=100)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=False)

    model = MLP(**model_args)
    model.initialize()

    # Train model for 2 epochs
    trainer = Trainer(model=model, **trainer_args)
    trainer.train(data_loader, 2)
    before = trainer.eval(data_loader)[0]

    # Save model
    rand = np.random.randint(10000, 99999)
    fn = f"models/model_{rand}.pt"
    model.save_model(fn)

    try:
        # Load model
        model = MLP(**model_args)
        model.load_model(fn)

        # Loss should be same before/after loading
        after = trainer.eval(data_loader)[0]
        assert np.isclose(before, after)
    finally:
        os.remove(fn)


def test_continue_training():
    # Train for ten epochs, save per-epoch losses
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    dataset = AddDataset(num_examples=100)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=False)

    model = MLP(**model_args)
    model.initialize()

    trainer = Trainer(model=model, **trainer_args)
    ten_losses = trainer.train(data_loader, 10)

    # Start over and train for just five epochs
    np.random.seed(0)
    torch.manual_seed(0)

    dataset = AddDataset(num_examples=100)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=False)

    model = MLP(**model_args)
    model.initialize()

    trainer = Trainer(model=model, **trainer_args)
    five_losses = trainer.train(data_loader, 5)

    try:
        # Save the model and trainer
        now = re.sub("[ :.-]", "_", str(datetime.datetime.now()))
        model_fn = f"models/test_model_{now}.pt"
        trainer_fn = f"models/test_trainer_{now}.pt"
        model.save_model(model_fn)
        trainer.save_trainer(trainer_fn)

        # Reload the model and the trainer
        model = MLP(**model_args)
        model.load_model(model_fn)
        trainer = Trainer(model=model, **trainer_args)
        trainer.load_trainer(trainer_fn)

        # Train for five more epochs
        np.random.seed(0)
        torch.manual_seed(0)
        five_more = trainer.train(data_loader, 5)

        # Loss should be the same whether you train once for ten
        #   epochs or twice for five epochs
        assert np.all(np.isclose(ten_losses, five_losses + five_more))
    finally:
        if os.path.exists(model_fn):
            os.remove(model_fn)
        if os.path.exists(trainer_fn):
            os.remove(trainer_fn)
