import random
import numpy as np
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
    # Add some food sources (2)
    maze[2, 3] = 4
    maze[3, 5] = 4
    maze[4, 7] = 4
    return maze

food_counter = 3

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

def all_food_found():
    return 
# Animation logic
def update(frame):
    global slime_coords
    global im
    global food_counter
    slime_coords = grow_slime(maze, slime_coords)
    # Update the displayed maze
    display_maze = maze.copy()
    
    if food_counter == 0:
        while True:
            display_maze[9,9] = 4
            im.set_data(display_maze)     
    for x, y in slime_coords:
        if display_maze[x, y] == 0 or display_maze[x, y] == 4:  # Only overwrite empty spaces
            display_maze[x, y] = 3  # Mark slime cells
        if display_maze[x, y] == 4:
            food_counter -= 1
    #maze = display_maze
    im.set_data(display_maze)

# Initialize the maze and slime
rows, cols = 10, 10
maze = create_maze(rows, cols)
slime_coords = [(5, 5)]  # Starting at the center

# Set up visualization
cmap = ListedColormap(["white", "black", "red", "yellow"])  # 0: White, 1: Black, 2: Red, 3: Yellow
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xticks([])
ax.set_yticks([])
im = ax.imshow(maze, cmap="viridis", origin="upper")
ax.set_title("Maze with Slime Growth")

# Create animation
ani = FuncAnimation(fig, update, frames=1000, interval=20)  # 30 frames, 500ms per frame
plt.show()
