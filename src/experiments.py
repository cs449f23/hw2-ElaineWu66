import torch


def params_add_dataset():
    """
    Choose the parameters you used to train your AddDataset model
    This will be used to load the model you saved.

    Returns
        model_args: a dictionary of arguments to be passed to MLP()
        trainer_args: a dictionary of arguments to be passed to Trainer()
    """

    model_args = {}
    raise NotImplementedError

    # Don't include 'model' or 'loss_func' here
    # Just "optimizer" and any necessary kwargs
    trainer_args = {}
    raise NotImplementedError

    return model_args, trainer_args


def params_multiply_dataset():
    """
    Choose the parameters you used to train your MultiplyDataset model
    This will be used to load the model you saved.

    Returns
        model_args: a dictionary of arguments to be passed to MLP()
        trainer_args: a dictionary of arguments to be passed to Trainer()
    """

    model_args = {}
    raise NotImplementedError

    # Don't include 'model' or 'loss_func' here
    # Just "optimizer" and any necessary kwargs
    trainer_args = {}
    raise NotImplementedError

    return model_args, trainer_args
