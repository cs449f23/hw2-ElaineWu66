import torch


def params_add_dataset():
    """
    Choose the parameters you used to train your AddDataset model
    This will be used to load the model you saved.

    Returns
        model_args: a dictionary of arguments to be passed to MLP()
        trainer_args: a dictionary of arguments to be passed to Trainer()
    """


    # raise NotImplementedError

    # Don't include 'model' or 'loss_func' here
    # Just "optimizer" and any necessary kwargs


    # Define the model arguments
    model_args = {
        'number_of_hidden_layers': 3,  
        'input_size': 2,               
        'hidden_size': 64,             
        'activation': torch.nn.ReLU()  
    }

    trainer_args = {
        'optimizer': torch.optim.Adam,  
        # 'optimizer_kwargs': {'lr': 0.001}, 
        # 'batch_size': 32,            
        # 'num_epochs': 100            
    }

    # raise NotImplementedError

    return model_args, trainer_args


def params_multiply_dataset():
    """
    Choose the parameters you used to train your MultiplyDataset model
    This will be used to load the model you saved.

    Returns
        model_args: a dictionary of arguments to be passed to MLP()
        trainer_args: a dictionary of arguments to be passed to Trainer()
    """

    model_args = {
        'number_of_hidden_layers': 3,  
        'input_size': 2,               
        'hidden_size': 64,            
        'activation': torch.nn.ReLU()  
    }

    trainer_args = {
        'optimizer': torch.optim.Adam,  
        # 'optimizer_kwargs': {'lr': 0.001},  
        # 'batch_size': 32,              
        # 'num_epochs': 100             
    }

    return model_args, trainer_args