import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Labyrinth Parameters
LABYRINTH_SIZE = (10, 10)
WALL = -1
FOOD = 1
EMPTY = 0
AGENT = 2  # For visualizing the slime agent

# Create the labyrinth
def generate_labyrinth(size):
    labyrinth = np.zeros(size, dtype=int)
    for x in range(size[0]):
        for y in range(size[1]):
            if random.random() < 0.2:  # 20% chance of being a wall
                labyrinth[x, y] = WALL
    return labyrinth

# Place food
def place_food(labyrinth, food_positions):
    labyrinth[labyrinth == FOOD] = EMPTY  # Clear old food
    for x, y in food_positions:
        labyrinth[x, y] = FOOD

# Slime Agent
class SlimeAgent:
    def __init__(self, labyrinth, start_position, exploration_rate=0.1):
        self.labyrinth = labyrinth
        self.start_position = start_position
        self.exploration_rate = exploration_rate
        self.reset()

    def reset(self):
        self.positions = [self.start_position]
        self.food_collected = 0
        self.frontier = [self.start_position]
        self.visited = set([self.start_position])
        self.score = 0  # Efficiency score
        self.collect_food_positions()

    def collect_food_positions(self):
        self.reachable_food = set(
            (x, y) for x in range(LABYRINTH_SIZE[0]) for y in range(LABYRINTH_SIZE[1]) if self.labyrinth[x, y] == FOOD
        )

    def heuristic(self, position):
        if not self.reachable_food:
            return float('inf')
        x, y = position
        return min(abs(x - fx) + abs(y - fy) for fx, fy in self.reachable_food)

    def expand_one_step(self):
        new_frontier = []
        for current_pos in self.frontier:
            x, y = current_pos
            directions = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
            random.shuffle(directions)  # Randomize directions to explore

            for nx, ny in directions:
                if 0 <= nx < LABYRINTH_SIZE[0] and 0 <= ny < LABYRINTH_SIZE[1]:
                    if (nx, ny) not in self.visited and self.labyrinth[nx, ny] != WALL:
                        # Introduce exploration
                        if random.random() < self.exploration_rate:
                            continue
                        new_frontier.append((nx, ny))
                        self.visited.add((nx, ny))
                        self.score += 1  # Penalize slime expansion
                        if self.labyrinth[nx, ny] == FOOD:
                            self.food_collected += 1
                            self.labyrinth[nx, ny] = EMPTY  # Consume food
                            self.score -= 10  # Reward food collection

        # Sort the new frontier based on proximity to food
        new_frontier.sort(key=self.heuristic)
        self.frontier = new_frontier

    def expand_until_food_found(self):
        while not self.has_collected_all_food() and self.frontier:
            self.expand_one_step()

    def has_collected_all_food(self):
        return self.reachable_food.issubset(self.visited)

    def display_labyrinth(self):
        display_labyrinth = np.copy(self.labyrinth)
        for x, y in self.visited:
            display_labyrinth[x, y] = AGENT
        return display_labyrinth

# Training function
def train_agent(labyrinth, epochs):
    

    # Initial positions for food and slime
    empty_cells = [(x, y) for x in range(LABYRINTH_SIZE[0]) for y in range(LABYRINTH_SIZE[1]) if labyrinth[x, y] == EMPTY]
    initial_food_positions = random.sample(empty_cells, 5)
    initial_start_position = random.choice(empty_cells)
    place_food(labyrinth, initial_food_positions)

    # Initialize the agent
    agent = SlimeAgent(labyrinth, initial_start_position)
    best_score = float('inf')
    best_visited = set()
    best_generation = 0

    fig, ax = plt.subplots()
    ax.set_title("Agent Learning the Maze")

    generation = 0
    scores = []

    def update(frame):
        nonlocal generation, best_score, best_visited, best_generation

        # Expand slime until all food is found
        agent.expand_until_food_found()

        if agent.has_collected_all_food():
            scores.append(agent.score)
            generation += 1
            print(f"Generation {generation}: Score = {agent.score}, Visited = {len(agent.visited)}")

            # Check if this generation is the best so far
            if agent.score < best_score:
                best_score = agent.score
                best_visited = agent.visited.copy()
                best_generation = generation

            # Reset for the next generation
            place_food(labyrinth, initial_food_positions)
            agent.reset()
            # Slightly increase exploration to avoid getting stuck
            agent.exploration_rate = max(0.05, agent.exploration_rate * 0.95)

        # Update visualization
        ax.imshow(agent.display_labyrinth(), cmap="viridis")
        ax.set_title(f"Generation: {generation} | Score: {agent.score} | Best: {best_score} (Gen {best_generation})")

    ani = FuncAnimation(fig, update, frames=epochs, repeat=False)
    plt.show()

    # Plot efficiency scores over generations
    plt.figure()
    plt.plot(scores, label="Scores")
    plt.axhline(y=best_score, color='r', linestyle='--', label=f"Best Score (Gen {best_generation})")
    plt.legend()
    plt.title("Efficiency Across Generations")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.show()

labyrinth = generate_labyrinth(LABYRINTH_SIZE)
train_agent(labyrinth, epochs=100)
