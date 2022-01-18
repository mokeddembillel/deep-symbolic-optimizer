import torch as T
import torch.nn as nn




class Actor(nn.Module):
    def __init__(self, lr, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        
        
        # Initialize layers
        self.lstm1 = nn.LSTM(self.input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim, self.output_dim)
        
        
        # Set learning rate
        self.lr = lr
        # Define optimizer
        self.optimizer = T.optim.Adam(self.parameters(), lr=self.lr)
        # Set Device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, input, hidden):
        output, hidden = self.lstm1(input, hidden)
        output = T.sigmoid(self.fc1(output))
        return output, hidden
        
    
        