import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from game import Game  # Assuming this is the correct import from your game environment
import matplotlib.pyplot as plt


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
        self.conv1_1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1)
        # self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        # self.conv1 = nn.Conv2d(1, 128, kernel_size=(2, 2))
        # self.conv2 = nn.Conv2d(128, 128, kernel_size=(2, 2))


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
        x = torch.cat((x1, x2), 1) 
        return x
        # x1 = torch.relu(self.conv1(x))
        # x2 = torch.relu(self.conv2(x1))
        # x3 = torch.relu(self.conv3(x2))
        # x4 = torch.relu(self.conv4(x3))
        # return x4
        # x = self.conv1(x)
        # x = self.conv2(x)
        # return x

    def forward(self, x):
        x = self._forward_conv(x)
        # x = self.dropout(x) potentially add later
        x = x.view(x.size(0), -1)  # Flatten the tensor
        return self.fc(x)

def epsilon_greedy_action(Q_values, epsilon=0.05):
    if random.random() < epsilon:
        # Randomly select an action
        return random.randint(0, len(Q_values) - 1)
    else:
        # Select the action with the highest Q-value
        return np.argmax(Q_values)

def train_game(game, it, batch_size, gamma, optimizer, criterion, device, model):
    global losses
    global scores  # Introducing a new variable to accumulate scores
    batch_outputs = []  # Tensors list for outputs
    batch_labels = []  # Tensors list for labels
    step = 1

    while not game.game_over():
        state = one_hot_encode_game_state(game.state())  # convert into one-hot
        state_tensor = state.unsqueeze(0).permute(0, 3, 1, 2).to(device)

        Q_values = model(state_tensor)
        Q_valid_values = [Q_values[0][a] if game.is_action_available(a) else float('-inf') for a in range(4)]
        action = epsilon_greedy_action(np.array(Q_valid_values))
        # best_action = np.argmax(np.array(Q_valid_values))
        # reward = game.do_action(best_action)
        reward = game.do_action(action)
        # Q_star = Q_values[0][best_action]  # Directly use the Q value from the tensor
        Q_star = Q_values[0][action]  # Directly use the Q value from the tensor

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
                scores.append(game.score())  # Collect score
                if it % 100 == 0 and it > 0:  # Every 100 epochs, calculate and print the mean score
                    mean_score = sum(scores[-100:]) / 100
                    print("Epoch: {}, Mean score last 100 epochs: {:.2f}".format(it, mean_score))
                # print("Epoch: {}, Game score: {}".format(it, game.score()))
                return
            #game.print_state()
        step += 1

def eval_game(game, n_eval, device, model):
    total_score = 0

    for _ in range(n_eval):
        game = Game()
        while not game.game_over():
            state = one_hot_encode_game_state(game.state())  # convert into one-hot
            state_tensor = state.unsqueeze(0).permute(0, 3, 1, 2).to(device)

            with torch.no_grad():
                Q_values = model(state_tensor)
                Q_valid_values = [Q_values[0][a] if game.is_action_available(a) else float('-inf') for a in range(4)]
                best_action = np.argmax(Q_valid_values)
            
            game.do_action(best_action)

        total_score += game.score()  # Sum up the scores after each game ends

    mean_score = total_score / n_eval  # Calculate the mean score over all evaluated games
    return mean_score

if __name__=="__main__":
    input_shape = (16, 4, 4)  # Replace with the actual shape of the game state
    num_actions = 4  # Replace with the actual number of actions in the game
    batch_size = 128
    gamma = 1  # Discount factor
    n_epoch = 1500
    n_eval = 100
    SEED = 1

    # Set random seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net(input_shape, num_actions).to(device)
    learning_rate = 0.00001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss().to(device)

    losses = []
    scores = []
    randoms = []
    highest_tiles = []
    game = Game()
    
    model.train()
    for it in range(n_epoch):
        game = Game()  # Replace with actual game initialization
        train_game(game, it, batch_size, gamma, optimizer, criterion, device, model)
        highest_tiles.append(game.max_tile())
        
    test_games = 1000
    mean_score_eval = eval_game(game, test_games, device, model)
    print(f"The mean of the scores across {test_games} evaluation games is {mean_score_eval}")

    # After the training loop
    plt.plot(scores)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Training Scores over Epochs')
    plt.show()  
    plt.close()

    plt.scatter(range(len(highest_tiles)), highest_tiles, label='Highest Tile', marker='o', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Tile Value')
    plt.title('Highest Tile Reached over Epochs')
    plt.legend()
    plt.show()
    plt.close()