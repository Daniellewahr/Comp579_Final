import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from game import Game  # Assuming this is the correct import from your game environment


def one_hot_encode_game_state(game_state):
    """
    Convert a 4x4 matrix of integers with values 1-16 to a 4x4x16 matrix of one-hot encoded vectors.
    
    Parameters:
    game_state (np.ndarray): a 4x4 matrix with integer values in range [1, 16].
    
    Returns:
    torch.Tensor: a 4x4x16 tensor of one-hot encoded vectors.
    """
    # Initialize a tensor of zeros with the shape 4x4x16
    one_hot_encoded_state = torch.zeros(game_state.shape + (16,))
    
    # Populate the tensor with one-hot encoding
    for i in range(game_state.shape[0]):
        for j in range(game_state.shape[1]):
            # Subtract 1 from the value to get zero-based index for one-hot encoding
            index = game_state[i, j] - 1
            one_hot_encoded_state[i, j, index] = 1
    
    return one_hot_encoded_state


class Net(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(Net, self).__init__()
        # Ensuring that both convolutional paths produce the same output dimensions
        self.conv1_1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1)

        # Calculate the correct size after convolution without assuming manual sizes
        linear_input_size = self._get_conv_output(input_shape)
        self.fc = nn.Linear(linear_input_size, num_actions)

    def _get_conv_output(self, shape):
        input = torch.rand(1, *shape)
        output = self._forward_conv(input)
        return int(np.prod(output.size()))

    def _forward_conv(self, x):
        x1 = torch.relu(self.conv1_1(x))
        x2 = torch.relu(self.conv1_2(x))
        x = torch.cat((x1, x2), 1)  # Should now be able to concatenate without error
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        # x = self.dropout(x) potentially add later
        x = x.view(x.size(0), -1)  # Flatten the tensor
        return self.fc(x)

# Assuming game.state() returns the current state in the correct shape for the CNN
# and game.vector() is no longer needed.
def train_game(game, it, batch_size, gamma, optimizer, criterion, device):
    global losses
    batch_outputs = []  # Tensors list for outputs
    batch_labels = []  # Tensors list for labels
    step = 1
    while not game.game_over():
        state = one_hot_encode_game_state(game.state())  # convert into one-hot
        state_tensor = state.unsqueeze(0).permute(0, 3, 1, 2).to(device)

        Q_values = model(state_tensor)
        Q_valid_values = [Q_values[0][a] if game.is_action_available(a) else float('-inf') for a in range(4)]
        best_action = np.argmax(Q_valid_values)
        reward = game.do_action(best_action)
        Q_star = Q_values[0][best_action]  # Directly use the Q value from the tensor

        new_state = game.state()
        new_state_tensor = one_hot_encode_game_state(new_state).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        
        with torch.no_grad():
            Q_next = model(new_state_tensor)

        batch_outputs.append(Q_star)
        batch_labels.append(reward + gamma * torch.max(Q_next).item())  # Append directly as items

        if step % batch_size == 0 or game.game_over():
            if len(batch_labels) == 0: return
            optimizer.zero_grad()
            label_tensor = torch.tensor(batch_labels, dtype=torch.float32).to(device)
            output_tensor = torch.stack(batch_outputs).to(device)  # Stack tensors directly
            batch_labels, batch_outputs = [], []
            loss = criterion(output_tensor, label_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if game.game_over():
                print("epoch: {}, game score: {}".format(it, game.score()))
                return
            game.print_state()
        step += 1





def eval_game(game, n_eval, device):
    global scores
    model.eval()
    with torch.no_grad():
        for i in range(n_eval):
            game.reset()
            while not game.game_over():
                state = game.state()
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
                Q_values = model(state_tensor)
                Q_valid_values = [Q_values[0][a] if game.is_action_available(a) else float('-inf') for a in range(4)]
                best_action = np.argmax(Q_valid_values)
                game.do_action(best_action)
            scores.append(game.score())
            print("Game #{} score: {}".format(i+1, game.score()))





if __name__=="__main__":
    input_shape = (16, 4, 4)  # Replace with the actual shape of the game state
    num_actions = 4  # Replace with the actual number of actions in the game
    batch_size = 128
    gamma = 1  # Discount factor
    n_epoch = 1000
    n_eval = 100
    SEED = 1234

    # Set random seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net(input_shape, num_actions).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss().to(device)

    losses = []
    scores = []
    randoms = []

    

    model.train()
    for it in range(n_epoch):
        game = Game()  # Replace with actual game initialization
        train_game(game, it, batch_size, gamma, optimizer, criterion, device)
    
    eval_game(game, n_eval, device)
    print("The mean of the scores is {}".format(np.mean(scores)))