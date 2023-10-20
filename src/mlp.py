import torch
import torch.nn.init as init

from src.utils import save, load


class MLP(torch.nn.Module):
    def __init__(self, number_of_hidden_layers: int, input_size: int,
                 hidden_size: int, activation: torch.nn.Module):
        """
        Construct a simple MLP
        """
        # NOTE: don't edit this constructor

        super().__init__()
        assert number_of_hidden_layers >= 0, "number_of_hidden_layers must be at least 0"

        dims_in = [input_size] + [hidden_size] * number_of_hidden_layers
        dims_out = [hidden_size] * number_of_hidden_layers + [1]

        layers = []
        for i in range(number_of_hidden_layers + 1):
            layers.append(torch.nn.Linear(dims_in[i], dims_out[i]))

            # No final activation
            if i < number_of_hidden_layers:
                layers.append(activation)

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def initialize(self):
        """
        Initialize all the model's weights.
        See https://pytorch.org/docs/stable/nn.init.html
        """
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)        
        # raise NotImplementedError


    def save_model(self, filename):
        """
        Use `src.utils.save` to save this model to file.

        Note: You may want to save a dictionary containing the model's state.

        Args
            filename: the file to which to save the model
        """
        model_state = {
            'state_dict': self.state_dict(), 
        }

        # print("Model's state_dict:")
        # for param_tensor in model_state['state_dict']:
        #     print(param_tensor, "\t", model_state['state_dict'][param_tensor].size())

        save(filename,**model_state)
        # raise NotImplementedError

    def load_model(self, filename):
        """
        Use `src.utils.load` to load this model from file.

        Note: in addition to simply loading the saved model, you must use the
              information from that checkpoint to update the model's state.

        Args
            filename: the file from which to load the model
        """
        model_state = load(filename)
    
        self.load_state_dict(model_state['state_dict'])
        self.eval()
        # raise Not ImplementedError
