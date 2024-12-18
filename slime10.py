import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Labyrinth Parameters
LABYRINTH_SIZE = (20, 20)
WALL = -1
FOOD = 1
EMPTY = 0
AGENT = 2  # For visualizing the slime agent

# Create the labyrinth
def generate_labyrinth(size):
    labyrinth = np.zeros(size, dtype=int)
    for x in range(size[0]):
        for y in range(size[1]):
            if random.random() < 0.1:  # 10% chance of being a wall
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
        self.q_table = {}  # State-action memory
        self.reset()

    def reset(self):
        """Reset the agent for a new generation."""
        self.positions = [self.start_position]
        self.food_collected = 0
        self.frontier = [self.start_position]
        self.visited = set([self.start_position])
        self.score = 0  # Reset efficiency score
        self.collect_food_positions()

    def collect_food_positions(self):
        self.reachable_food = set(
            (x, y)
            for x in range(LABYRINTH_SIZE[0])
            for y in range(LABYRINTH_SIZE[1])
            if self.labyrinth[x, y] == FOOD
        )

    def has_collected_all_food(self):
        return all(food in self.visited for food in self.reachable_food)

    def display_labyrinth(self):
        """Create a visualization of the labyrinth showing the slime."""
        display_labyrinth = np.copy(self.labyrinth)
        for x, y in self.visited:
            display_labyrinth[x, y] = AGENT
        return display_labyrinth

    # Other methods like heuristic, get_q_value, update_q_value, choose_action, expand_one_step...


    def heuristic(self, position):
        if not self.reachable_food:
            return float('inf')
        x, y = position
        return min(abs(x - fx) + abs(y - fy) for fx, fy in self.reachable_food)

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, reward, learning_rate=0.1, discount_factor=0.9):
        old_value = self.get_q_value(state, action)
        best_future_value = max(
            self.get_q_value(state, next_action) for next_action in self.get_possible_actions(state)
        )
        self.q_table[(state, action)] = old_value + learning_rate * (
            reward + discount_factor * best_future_value - old_value
        )

    def get_possible_actions(self, state):
        x, y = state
        actions = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < LABYRINTH_SIZE[0] and 0 <= ny < LABYRINTH_SIZE[1]:
                if self.labyrinth[nx, ny] != WALL:
                    actions.append((dx, dy))
        return actions

    def choose_action(self, state):
        actions = self.get_possible_actions(state)
        if random.random() < self.exploration_rate:
            return random.choice(actions)
        # Exploit: Choose the action with the highest Q-value
        q_values = [(action, self.get_q_value(state, action)) for action in actions]
        return max(q_values, key=lambda x: x[1])[0]

    def expand_one_step(self):
        """Expand the slime by one step."""
        if not self.frontier:
            return  # Stop if no positions left to expand

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

        # Update the frontier with new positions
        self.frontier = new_frontier

    def expand_until_food_found(self):
        """Keep expanding until all food is collected or no more moves are possible."""
        while not self.has_collected_all_food() and self.frontier:
            self.expand_one_step()


# Training function
def train_agent(labyrinth, epochs):
    # Generate the initial labyrinth and food positions
    labyrinth = generate_labyrinth(LABYRINTH_SIZE)
    empty_cells = [(x, y) for x in range(LABYRINTH_SIZE[0]) for y in range(LABYRINTH_SIZE[1]) if labyrinth[x, y] == EMPTY]
    initial_food_positions = random.sample(empty_cells, 5)
    initial_start_position = random.choice(empty_cells)
    place_food(labyrinth, initial_food_positions)

    # Initialize the agent
    agent = SlimeAgent(labyrinth, initial_start_position)

    # Tracking best results and scores
    best_score = float('inf')  # Best score so far
    best_generation = 0  # Generation where the best score occurred
    best_visited = set()  # Best visited cells
    generation = 0  # Current generation counter
    scores = []  # List to store scores over generations

    fig, ax = plt.subplots()
    ax.set_title("Agent Learning the Maze")

    def update(frame):
        nonlocal generation, best_score, best_generation, best_visited

        try:
            # Expand slime until all food is found
            if not agent.has_collected_all_food():
                agent.expand_until_food_found()

            # If all food is collected, proceed to the next generation
            if agent.has_collected_all_food():
                scores.append(agent.score)
                print(f"Generation {generation + 1}: Score = {agent.score}, Visited = {len(agent.visited)}")

                # Update best score and visited set if applicable
                if agent.score < best_score:
                    best_score = agent.score
                    best_visited = agent.visited.copy()
                    best_generation = generation + 1

                # Reset for the next generation
                generation += 1
                agent.reset()
                place_food(labyrinth, initial_food_positions)
                agent.exploration_rate = max(0.05, agent.exploration_rate * 0.95)  # Reduce exploration

            # Update visualization
            ax.imshow(agent.display_labyrinth(), cmap="viridis")
            ax.set_title(f"Generation: {generation + 1} | Score: {agent.score} | Best: {best_score} (Gen {best_generation})")

        except Exception as e:
            print(f"Error in generation {generation + 1}: {e}")
            ani.event_source.stop()  # Stop the animation on error

    ani = FuncAnimation(fig, update, frames=epochs, repeat=False)
    plt.show()

    # Plot efficiency scores after training
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
