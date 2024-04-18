from DQN_CNN import *
from collections import namedtuple
from replay_buffer import ReplayBuffer
from per import PrioritizedReplayMemory

# uses a double dqn 
# prioritized experience replay
# If the game environment is complex or has a large state space, it can take longer for the agent to explore and learn optimal policies.

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state'])
# Create an instance of prioritized replay memory
capacity = 10000  # Adjust capacity as needed
per_memory = PrioritizedReplayMemory(capacity)


def train_game_with_per(game, it, batch_size, gamma, optimizer, criterion, device):
    global losses
    global scores
    batch_outputs = []
    batch_labels = []
    step = 1

    while not game.game_over():
        state = one_hot_encode_game_state(game.state())
        state_tensor = state.unsqueeze(0).permute(0, 3, 1, 2).to(device)

        Q_values = model(state_tensor)
        Q_valid_values = [Q_values[0][a] if game.is_action_available(a) else float('-inf') for a in range(4)]
        action = epsilon_greedy_action(np.array(Q_valid_values))
        reward = game.do_action(action)
        Q_star = Q_values[0][action]

        new_state = game.state()
        new_state_tensor = one_hot_encode_game_state(new_state).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        
        per_memory.push(state_tensor, action, reward, new_state_tensor)

        if len(per_memory) > batch_size:
            indices, batch, weights = per_memory.sample(batch_size)
            states, actions, rewards, next_states = zip(*batch)

            states = torch.cat(states)
            next_states = torch.cat(next_states)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            actions = torch.tensor(actions).to(device)

            # Calculate Q-values and actions for the next states using both networks
            Q_next_model = model(next_states)
            # Q_next_target = target_model(next_states)
            Q_next_best_actions = Q_next_model.max(1)[1].unsqueeze(1)
            Q_targets = rewards + gamma * Q_next_target.gather(1, Q_next_best_actions).squeeze(1)

            # Calculate Q-values for the current states using the primary network
            Q_values = model(states)
            Q_values = Q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Calculate TD errors and update priorities
            td_errors = Q_values - Q_targets
            per_memory.update_priorities(indices, td_errors.detach().cpu().numpy())

            # Compute the loss and perform optimization
            optimizer.zero_grad()
            loss = torch.mean((torch.tensor(weights) * td_errors) ** 2)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if game.game_over():
            scores.append(game.score())
            if it % 100 == 0 and it > 0:
                mean_score = sum(scores[-100:]) / 100
                print("Epoch: {}, Mean score last 100 epochs: {:.2f}".format(it, mean_score))
            return
        step += 1

def eval_game(game, n_eval, device):
    global model
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
    input_shape = (16, 4, 4)
    num_actions = 4
    batch_size = 32
    gamma = 1  # Discount factor
    n_epoch = 1000
    n_eval = 1000
    capacity = 10000
    epsilon = 1e-5

    SEED = 1
    # Set random seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a separate target network with the same architecture as the main network
    # model = Net(input_shape, num_actions).to(device)
    # for double dqn
    model = Net(input_shape, num_actions).to(device)
    # target_model.load_state_dict(model.state_dict())  # Initialize target network with the same weights as the main network
    update_target_freq = 100  # Update target network every 100 epochs (you can adjust this value)

    learning_rate = 0.00001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss().to(device)
    replay_buffer = ReplayBuffer(capacity)

    losses = []
    scores = []
    randoms = []
    highest_tiles = []
    game = Game()

    model.train()
    for it in range(n_epoch):
        game = Game()
        train_game_with_per(game, it, batch_size, gamma, optimizer, criterion, device)
        # if it % update_target_freq == 0:
        #     target_model.load_state_dict(model.state_dict())
        highest_tiles.append(game.max_tile())

    model.eval()
    mean_score_eval = eval_game(game, n_eval, device)
    print(f"The mean of the scores across {n_eval} games is {mean_score_eval}")

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

