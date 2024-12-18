import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

# Define the grid environment
def create_maze(rows, cols):
    """Create a grid-based maze with walls and food sources."""
    maze = np.zeros((rows, cols), dtype=int)
    # Add walls (1)
    maze[1, :] = 1
    maze[-2, :] = 1
    maze[:, 1] = 1
    maze[:, -2] = 1
    # Add some food sources (4)
    maze[2, 3] = 4
    maze[3, 5] = 4
    maze[4, 7] = 4
    return maze

# Growth logic
def get_valid_growth_positions(maze, slime_coords):
    """Get all valid growth positions from the current slime cells."""
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    valid_positions = []
    
    for x, y in slime_coords:
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and (maze[nx, ny] == 0 or maze[nx, ny] == 4):
                valid_positions.append((nx, ny))
    
    return valid_positions

def grow_slime(maze, slime_coords):
    """Randomly grow the slime by adding one new cell."""
    valid_positions = get_valid_growth_positions(maze, slime_coords)
    if valid_positions:
        new_cell = random.choice(valid_positions)
        slime_coords.append(new_cell)
    return slime_coords

# Neural Network definition
class SlimeNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SlimeNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Training loop
# Training loop with debugging output
def train(model, maze, num_generations=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for generation in range(num_generations):
        print(f"Starting generation {generation + 1}/{num_generations}")  # Debug print
        slime_coords = [(5, 5)]  # Start at the center of the maze
        visited = set(slime_coords)
        food_found = 0
        display_maze = maze.copy()

        # Run until all food is collected
        while food_found < 3:
            input_data = torch.FloatTensor(display_maze.flatten()).unsqueeze(0)  # Flatten the maze for the NN
            with torch.no_grad():
                action_scores = model(input_data)
            action = torch.argmax(action_scores).item()  # Get action with the highest score

            # Directions corresponding to actions: 0: up, 1: down, 2: left, 3: right
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            dx, dy = directions[action]
            x, y = slime_coords[-1]
            nx, ny = x + dx, y + dy

            # Check if the new position is valid (within bounds, empty, or food)
            if (
                0 <= nx < 10
                and 0 <= ny < 10
                and (display_maze[nx, ny] == 0 or display_maze[nx, ny] == 4)  # Can grow to empty space or food
                and (nx, ny) not in visited  # Don't revisit previous cells
            ):
                slime_coords.append((nx, ny))  # Add new slime cell
                visited.add((nx, ny))  # Mark the new cell as visited
            
                if display_maze[nx, ny] == 4:  # If food is found
                    food_found += 1  # Increase food found counter
                    display_maze[nx, ny] = 0  # Remove food from the maze (turn to empty space)
            print(display_maze[nx, ny])
            print("X = " + str(nx) + " Y = " + str(ny))
            # Allow slime to explore further if it's still finding food
            if food_found == 3:
                break

        # Calculate reward: Smaller slime size = higher reward
        size_penalty = len(slime_coords)  # Reward for smaller slime
        reward = max(100 - size_penalty, 0)  # Reward decreases with size

        # Target for training: Higher reward for smaller slime sizes
        target = torch.FloatTensor([reward])

        optimizer.zero_grad()
        output = model(torch.FloatTensor(display_maze.flatten()).unsqueeze(0))  # Predict value from NN
        loss = criterion(output, target)  # Calculate the loss
        loss.backward()  # Backpropagate
        optimizer.step()  # Update weights

        print(f"Generation {generation + 1}: Loss = {loss.item()}, Reward = {reward}")  # Debug print

        # Early stop if the loss is very small (indicating convergence)
        if loss.item() < 0.01:
            print("Training converged, loss is very small.")
            break

        if generation % 10 == 0:
            print(f"Generation {generation}/{num_generations}, Loss: {loss.item()}, Reward: {reward}")

# Create the environment
rows, cols = 10, 10
maze = create_maze(rows, cols)

# Run the training
model = SlimeNN(input_size=rows * cols, output_size=4)  # Maze is 10x10, 4 possible actions
print("Starting training...")
train(model, maze)  # Ensure this call is made



# Setup for visualization
cmap = ListedColormap(["white", "black", "red", "yellow"])  # 0: White, 1: Black, 2: Red (food), 3: Yellow (slime)
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xticks([])
ax.set_yticks([])
im = ax.imshow(maze, cmap=cmap, origin="upper")
ax.set_title("Maze with Slime Growth")

# Grow the slime using the trained model
def grow_slime_with_nn_visualization(maze, slime_coords, model):
    """Generator for slime growth visualization."""
    food_found = 0
    visited = set(slime_coords)
    display_maze = maze.copy()

    while food_found < 3:  # Continue until all food is found
        input_data = torch.FloatTensor(display_maze.flatten()).unsqueeze(0)
        with torch.no_grad():
            action_scores = model(input_data)
        action = torch.argmax(action_scores).item()

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = directions[action]
        x, y = slime_coords[-1]
        nx, ny = x + dx, y + dy

        if (
            0 <= nx < rows
            and 0 <= ny < cols
            and (display_maze[nx, ny] == 0 or display_maze[nx, ny] == 4)
            and (nx, ny) not in visited
        ):
            slime_coords.append((nx, ny))
            visited.add((nx, ny))
            if display_maze[nx, ny] == 4:  # Food found
                food_found += 1
                display_maze[nx, ny] = 0  # Remove food
        else:
            break  # If no valid move, break out of the loop.

        # Yield the updated maze and slime coordinates for animation
        yield display_maze.copy(), slime_coords.copy()

    yield display_maze.copy(), slime_coords.copy()  # Final state

# Update function for animation
def update(frame_data):
    """Update the animation frame."""
    display_maze, slime_coords = frame_data
    # Mark slime cells with yellow (3)
    for x, y in slime_coords:
        display_maze[x, y] = 3  # Yellow for slime

    im.set_data(display_maze)

# Final Visualization using the trained model
def visualize_final_slime(model):
    """Run the final animation using the trained model."""
    maze = create_maze(rows, cols)
    slime_coords = [(5, 5)]  # Starting position
    generator = grow_slime_with_nn_visualization(maze, slime_coords, model)

    ani = FuncAnimation(
        fig,
        update,
        frames=generator,
        interval=200,
        repeat=False,
        cache_frame_data=False,  # Disable caching of frames
    )
    
    plt.show()  # Make sure to call plt.show() to start the animation

# Training the model
model = SlimeNN(input_size=rows * cols, output_size=4)  # Maze is 10x10, 4 possible actions
train(model, maze)

# After training is complete, run the final slime visualization
visualize_final_slime(model)
