import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import defaultdict

# Labyrinth parameters
LABYRINTH_SIZE = (20, 20)
WALL = -1
FOOD = 1
EMPTY = 0
AGENT = 2

# Q-learning parameters
ALPHA = 0.2       # Increased
GAMMA = 0.95      # Increased
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.995  # Slower decay

ACTIONS = [(0,1), (0,-1), (1,0), (-1,0)]  # Right, Left, Down, Up

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

class SlimeAgent:
    def __init__(self, labyrinth, start_position, total_food, epsilon=INITIAL_EPSILON):
        self.labyrinth = labyrinth
        self.start_position = start_position
        self.position = start_position
        self.epsilon = epsilon
        self.total_food = total_food
        self.food_remaining = total_food
        self.reset_stats()

    def reset_stats(self):
        self.position = self.start_position
        self.food_collected = 0
        self.food_remaining = self.total_food
        self.steps = 0
        self.score = 0
        self.visited = set([self.start_position])

    def get_state(self):
        # State includes agent’s x, y, and how many foods remain
        return (self.position[0], self.position[1], self.food_remaining)

    def choose_action(self, Q):
        state = self.get_state()
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        else:
            q_values = [Q[(state, a)] for a in ACTIONS]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(ACTIONS, q_values) if q == max_q]
            return random.choice(best_actions)

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
            # Extra bonus if all food collected
            if self.food_remaining == 0:
                reward += 20
                done = True

        return new_pos, reward, done

    def display_labyrinth(self):
        display_lab = np.copy(self.labyrinth)
        for vx, vy in self.visited:
            display_lab[vx, vy] = AGENT
        return display_lab

def train_agent(epochs=50, max_steps=900, food_count=5):
    # Generate labyrinth once
    original_labyrinth = generate_labyrinth(LABYRINTH_SIZE)

    # Find empty cells for food and agent start
    empty_cells = [(x, y) for x in range(LABYRINTH_SIZE[0]) for y in range(LABYRINTH_SIZE[1]) 
                   if original_labyrinth[x, y] == EMPTY]
    initial_food_positions = random.sample(empty_cells, food_count)
    initial_start_position = random.choice(empty_cells)

    # Create a copy of the labyrinth to restore each generation
    initial_labyrinth = np.copy(original_labyrinth)
    place_food(initial_labyrinth, initial_food_positions)

    # Initialize Q-table and agent
    Q = defaultdict(float)
    agent = SlimeAgent(np.copy(initial_labyrinth), initial_start_position, total_food=food_count)

    best_score = -float('inf')
    best_generation = 0
    scores = []

    fig, ax = plt.subplots()
    fig.suptitle("Agent Learning the Maze", fontsize=16)

    generation = 0

    def run_generation(gen):
        agent.reset_stats()
        # Restore labyrinth
        current_labyrinth = np.copy(original_labyrinth)
        place_food(current_labyrinth, initial_food_positions)
        agent.labyrinth = current_labyrinth

        for step in range(max_steps):
            state = agent.get_state()
            action = agent.choose_action(Q)
            new_pos, reward, done = agent.step_environment(action)

            # Q-learning update
            next_state = (new_pos[0], new_pos[1], agent.food_remaining)
            old_value = Q[(state, action)]
            future_values = [Q[(next_state, a)] for a in ACTIONS]
            Q[(state, action)] = old_value + ALPHA * (
                reward + GAMMA * max(future_values) - old_value
            )

            # Update agent stats
            agent.position = new_pos
            agent.visited.add(new_pos)
            agent.score += reward
            agent.steps += 1

            if done:
                break

        # Decay epsilon
        agent.epsilon = max(MIN_EPSILON, agent.epsilon * EPSILON_DECAY)

        nonlocal best_score, best_generation
        if agent.score > best_score:
            best_score = agent.score
            best_generation = gen

        scores.append(agent.score)
        return agent.display_labyrinth(), agent.score

    def update(frame):
        nonlocal generation
        display_lab, score = run_generation(frame)
        generation += 1
        ax.clear()
        ax.imshow(display_lab, cmap="viridis")
        ax.set_title(
            f"Gen: {generation} | Score: {score:.2f} | "
            f"Best: {best_score:.2f} (Gen {best_generation})"
        )
        return [ax]

    ani = FuncAnimation(fig, update, frames=epochs, repeat=False)
    plt.show()

    # After training, plot the scores
    plt.figure()
    plt.plot(scores, label="Score per Generation")
    plt.axhline(y=best_score, color='r', linestyle='--',
                label=f"Best Score (Gen {best_generation})")
    plt.title("Performance Across Generations")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

# Run the training
train_agent(epochs=500, max_steps=9000, food_count=9)