import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Hyperparameters
LABYRINTH_SIZE = (20, 20)
WALL = -1
FOOD = 1
EMPTY = 0
AGENT = 2
ALPHA = 0.001  # Learning rate for neural network
GAMMA = 0.95  # Discount factor
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 10000
TARGET_UPDATE_INTERVAL = 10

ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

def generate_labyrinth(size):
    labyrinth = np.zeros(size, dtype=int)
    for x in range(size[0]):
        for y in range(size[1]):
            if random.random() < 0.1:  # 10% chance of wall
                labyrinth[x, y] = WALL
    return labyrinth

def place_food(labyrinth, food_positions):
    labyrinth[labyrinth == FOOD] = EMPTY
    for x, y in food_positions:
        labyrinth[x, y] = FOOD
        

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


class DQNSlimeAgent:
    def display_labyrinth(self):
        """
        Generate a visual representation of the labyrinth, marking the agent's current position
        and visited cells.
        """
        display_lab = np.copy(self.labyrinth)
        for vx, vy in self.visited:
            display_lab[vx, vy] = AGENT
        return display_lab

    
    def reset_stats(self):
        """
        Reset the agent's stats for a new episode.
        """
        self.position = self.start_position
        self.food_collected = 0
        self.food_remaining = self.total_food
        self.steps = 0
        self.score = 0
        self.visited = set([self.start_position])  # Initialize as a set
    def __init__(self, labyrinth, start_position, total_food):
        self.labyrinth = labyrinth
        self.start_position = start_position
        self.position = start_position
        self.total_food = total_food
        self.food_remaining = total_food
        self.epsilon = INITIAL_EPSILON
        self.visited = False

        # Neural networks
        input_size = 3  # x, y, food_remaining
        output_size = len(ACTIONS)
        self.q_network = QNetwork(input_size, output_size)
        self.target_network = QNetwork(input_size, output_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=ALPHA)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Statistics
        self.reset_stats()

    def reset_stats(self):
        self.position = self.start_position
        self.food_collected = 0
        self.food_remaining = self.total_food
        self.steps = 0
        self.score = 0

    def get_state(self):
        return np.array([self.position[0], self.position[1], self.food_remaining], dtype=np.float32)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return ACTIONS[torch.argmax(q_values).item()]

    def step_environment(self, action):
        x, y = self.position
        nx, ny = x + action[0], y + action[1]

        # Default step reward is slightly negative
        reward = -0.1
        done = False

        # Check boundaries
        if not (0 <= nx < LABYRINTH_SIZE[0] and 0 <= ny < LABYRINTH_SIZE[1]):
            reward = -5  # penalty for out-of-bounds
            return self.position, reward, False

        # Check walls
        if self.labyrinth[nx, ny] == WALL:
            reward = -5  # penalty for hitting a wall
            return self.position, reward, False

        new_pos = (nx, ny)

        # If food is found
        if self.labyrinth[nx, ny] == FOOD:
            reward = 20
            self.labyrinth[nx, ny] = EMPTY
            self.food_collected += 1
            self.food_remaining -= 1
            if self.food_remaining == 0:  # All food collected
                reward += 20
                done = True

        # Update position and mark as visited
        self.visited.add(new_pos)
        return new_pos, reward, done

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def train(self):
        if self.replay_buffer.size() < BATCH_SIZE:
            return

        # Sample a batch of experiences
        batch = self.replay_buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert lists of NumPy arrays to single NumPy arrays
        states = np.array(states)
        next_states = np.array(next_states)

        # Convert to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor([ACTIONS.index(a) for a in actions], dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Compute Q-values and target Q-values
        q_values = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * GAMMA * next_q_values

        # Compute loss and update the Q-network
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

from tqdm import tqdm  # Progress bar
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def train_agent_with_animation(epochs=100, max_steps=500, food_count=5, animate_interval=10):
    labyrinth = generate_labyrinth(LABYRINTH_SIZE)
    empty_cells = [(x, y) for x in range(LABYRINTH_SIZE[0]) for y in range(LABYRINTH_SIZE[1])
                   if labyrinth[x, y] == EMPTY]
    initial_food_positions = random.sample(empty_cells, food_count)
    initial_start_position = random.choice(empty_cells)

    place_food(labyrinth, initial_food_positions)
    agent = DQNSlimeAgent(np.copy(labyrinth), initial_start_position, total_food=food_count)

    best_score = -float('inf')
    best_epoch = 0
    scores = []

    fig, ax = plt.subplots()
    fig.suptitle("Agent Learning the Maze", fontsize=16)

    def update_animation(frame):
        nonlocal best_score, best_epoch

        agent.reset_stats()
        current_labyrinth = np.copy(labyrinth)
        place_food(current_labyrinth, initial_food_positions)
        agent.labyrinth = current_labyrinth

        for step in range(max_steps):
            state = agent.get_state()
            action = agent.choose_action(state)
            new_pos, reward, done = agent.step_environment(action)

            next_state = np.array([new_pos[0], new_pos[1], agent.food_remaining], dtype=np.float32)
            agent.replay_buffer.add((state, action, reward, next_state, done))

            agent.position = new_pos
            agent.score += reward
            agent.steps += 1

            agent.train()

            if done:
                break

        if frame % TARGET_UPDATE_INTERVAL == 0:
            agent.update_target_network()

        agent.epsilon = max(MIN_EPSILON, agent.epsilon * EPSILON_DECAY)
        scores.append(agent.score)

        if agent.score > best_score:
            best_score = agent.score
            best_epoch = frame

        ax.clear()
        ax.imshow(agent.display_labyrinth(), cmap="viridis")
        ax.set_title(
            f"Epoch {frame} | Score: {agent.score:.2f} | "
            f"Best: {best_score:.2f} (Epoch {best_epoch})"
        )

    ani = FuncAnimation(fig, update_animation, frames=epochs, interval=animate_interval, repeat=False)
    plt.show()

    # Final Score Plot
    plt.figure()
    plt.plot(scores, label="Score per Epoch")
    plt.axhline(y=best_score, color='r', linestyle='--',
                label=f"Best Score (Epoch {best_epoch})")
    plt.title("Performance Across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

# Run the training with animation
train_agent_with_animation(epochs=200, max_steps=1000, food_count=10, animate_interval=100)

# Run the training
#train_agent()
