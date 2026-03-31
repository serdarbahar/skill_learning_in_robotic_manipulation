import torch

class MLP(torch.nn.Module):

    def __init__(self, input_dim,
                        hidden_dim,
                        output_dim,
                        num_hidden_layers,
                        activation_fn=torch.nn.ReLU):
            super(MLP, self).__init__()
    
            layers = []
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(activation_fn())
            for _ in range(num_hidden_layers):
                layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation_fn())
            layers.append(torch.nn.Linear(hidden_dim, output_dim))
    
            self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    
    