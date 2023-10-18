import numpy as np
import torch

from src.trainer import Trainer


class DummyModel(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weights = torch.nn.Linear(1, 1, bias=False)

    def forward(self, X):
        return self.weights(torch.ones([1, 1], dtype=torch.float32))


def test_trainer_basics():
    n = 100
    target = (torch.zeros(1, dtype=torch.float32),
              torch.zeros(1, dtype=torch.float32))

    data = [target for _ in range(n)]
    model = DummyModel(n)
    data_loader = torch.utils.data.DataLoader(data, batch_size=1)

    trainer = Trainer(
        optimizer=torch.optim.SGD,
        model=model,
        loss_func=torch.nn.MSELoss(),
        lr=0.1,
    )

    # Test `run_one_batch` with a dummy example
    loss_before = trainer.eval(data_loader)[0]
    trainer.run_one_batch(
        None, torch.zeros([1, 1], dtype=torch.float32))
    loss_after = trainer.eval(data_loader)[0]
    assert loss_after < loss_before, "Loss should decrease"

    # train=False
    loss_again = trainer.train(data_loader, 10, train=False, report_every=100)
    msg = "train=False should mean no training"
    assert np.all(np.isclose(loss_after, loss_again)), msg

    # Test trainer.train with train=True
    _ = trainer.train(data_loader, 3, report_every=100)
    losses = trainer.eval(data_loader)
    msg = "DummyModel should learn zero weights"
    assert np.isclose(losses[0], 0)

    model_weight = model.weights.weight[0][0].detach()
    assert np.isclose(model_weight, 0)
