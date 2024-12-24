# from rob831.hw4_part2.infrastructure import pytorch_util as ptu
# from .base_exploration_model import BaseExplorationModel
# import torch.optim as optim
# from torch import nn
# import torch

# def init_method_1(model):
#     model.weight.data.uniform_()
#     model.bias.data.uniform_()

# def init_method_2(model):
#     model.weight.data.normal_()
#     model.bias.data.normal_()


# class RNDModel(nn.Module, BaseExplorationModel):
#     def __init__(self, hparams, optimizer_spec, **kwargs):
#         super().__init__(**kwargs)
#         self.ob_dim = hparams['ob_dim']
#         self.output_size = hparams['rnd_output_size']
#         self.n_layers = hparams['rnd_n_layers']
#         self.size = hparams['rnd_size']
#         self.optimizer_spec = optimizer_spec

#         # <TODO>: Create two neural networks:
#         # 1) f, the random function we are trying to learn
#         # 2) f_hat, the function we are using to learn f

#     def forward(self, ob_no):
#         # <TODO>: Get the prediction error for ob_no
#         # HINT: Remember to detach the output of self.f!
#         pass

#     def forward_np(self, ob_no):
#         ob_no = ptu.from_numpy(ob_no)
#         error = self(ob_no)
#         return ptu.to_numpy(error)

#     def update(self, ob_no):
#         # <TODO>: Update f_hat using ob_no
#         # Hint: Take the mean prediction error across the batch
#         pass


from rob831.hw4_part2.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch

def init_method_1(model):
    if isinstance(model, nn.Linear):
        model.weight.data.uniform_()
        model.bias.data.uniform_()

def init_method_2(model):
    if isinstance(model, nn.Linear):
        model.weight.data.normal_()
        model.bias.data.normal_()

class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec

        # Create the random function f (target network)
        self.f = self._create_network()
        # Create the predictor network f_hat
        self.f_hat = self._create_network()
        
        # Initialize the networks differently
        self.f.apply(init_method_1)  # Initialize target network with uniform
        self.f_hat.apply(init_method_2)  # Initialize predictor network with normal
        
        # Create optimizer for f_hat
        self.optimizer = self.optimizer_spec.constructor(
            self.f_hat.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        
        # Move networks to correct device
        self.f = self.f.to(ptu.device)
        self.f_hat = self.f_hat.to(ptu.device)

    def _create_network(self):
        layers = []
        in_size = self.ob_dim
        
        # Create hidden layers
        for _ in range(self.n_layers):
            layers.append(nn.Linear(in_size, self.size))
            layers.append(nn.ReLU())
            in_size = self.size
        
        # Add output layer
        layers.append(nn.Linear(in_size, self.output_size))
        
        return nn.Sequential(*layers)

    def forward(self, ob_no):
        # Get predictions from both networks
        with torch.no_grad():
            target_features = self.f(ob_no)
        predicted_features = self.f_hat(ob_no)
        
        # Calculate prediction error (MSE for each sample)
        error = ((predicted_features - target_features) ** 2).mean(dim=1)
        return error

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)

    def update(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        prediction_error = self(ob_no)
        
        # Take mean prediction error for the batch
        loss = prediction_error.mean()
        
        # Update predictor network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()