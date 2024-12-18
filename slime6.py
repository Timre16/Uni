import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Labyrinth-Parameter
LABYRINTH_SIZE = (10, 10)
WALL = -1
FOOD = 1
EMPTY = 0
AGENT = 2  # Für die visuelle Darstellung des Agenten

# Labyrinth erstellen
def generate_labyrinth(size):
    labyrinth = np.zeros(size, dtype=int)
    # Wände zufällig generieren
    for x in range(size[0]):
        for y in range(size[1]):
            if random.random() < 0.2:  # 20% Wahrscheinlichkeit für eine Wand
                labyrinth[x, y] = WALL
    return labyrinth

# Essen platzieren
def place_food(labyrinth, food_count=5):
    # Alte Lebensmittel entfernen
    labyrinth[labyrinth == FOOD] = EMPTY
    # Neue Lebensmittel zufällig platzieren
    empty_cells = [(x, y) for x in range(labyrinth.shape[0]) for y in range(labyrinth.shape[1]) if labyrinth[x, y] == EMPTY]
    for _ in range(food_count):
        if empty_cells:
            x, y = random.choice(empty_cells)
            labyrinth[x, y] = FOOD
            empty_cells.remove((x, y))

# Agent-Umgebung
class SlimeAgent:
    def __init__(self, labyrinth):
        self.labyrinth = labyrinth
        self.reset()

    def reset(self):
        # Startposition zufällig setzen
        self.start_position = (random.randint(0, self.labyrinth.shape[0] - 1), 
                               random.randint(0, self.labyrinth.shape[1] - 1))
        while self.labyrinth[self.start_position] == WALL:  # Vermeide Start auf einer Wand
            self.start_position = (random.randint(0, self.labyrinth.shape[0] - 1), 
                                   random.randint(0, self.labyrinth.shape[1] - 1))
        self.positions = [self.start_position]
        self.food_collected = 0
        self.frontier = [self.start_position]
        self.visited = set([self.start_position])
        self.score = 0  # Bewertung der Effizienz
        self.collect_food_positions()

    def collect_food_positions(self):
        # Sammle alle Essenspositionen
        self.reachable_food = set()
        for x in range(LABYRINTH_SIZE[0]):
            for y in range(LABYRINTH_SIZE[1]):
                if self.labyrinth[x, y] == FOOD:
                    self.reachable_food.add((x, y))

    def heuristic(self, position):
        # Einfache Heuristik: Minimale Distanz zu einem Lebensmittel
        if not self.reachable_food:
            return float('inf')
        x, y = position
        return min(abs(x - fx) + abs(y - fy) for fx, fy in self.reachable_food)

    def expand_one_step(self):
        new_frontier = []
        for current_pos in self.frontier:
            x, y = current_pos
            directions = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
            random.shuffle(directions)  # Zufällige Richtung für Exploration

            for nx, ny in directions:
                if 0 <= nx < LABYRINTH_SIZE[0] and 0 <= ny < LABYRINTH_SIZE[1]:
                    if (nx, ny) not in self.visited and self.labyrinth[nx, ny] != WALL:
                        new_frontier.append((nx, ny))
                        self.visited.add((nx, ny))
                        self.score += 1  # Penalize expansion slightly
                        
                        # Reward for finding food
                        if self.labyrinth[nx, ny] == FOOD:
                            self.food_collected += 1
                            self.labyrinth[nx, ny] = EMPTY
                            self.score -= 20  # Reward heavily for food

        # Prioritize new frontier based on proximity to food
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

# Haupt-Trainingsfunktion
def train_agent(labyrinth, epochs=100):
    
    place_food(labyrinth, food_count=5)
    agent = SlimeAgent(labyrinth)

    fig, ax = plt.subplots()
    ax.set_title("Agent lernt das Labyrinth")

    generation = 0
    scores = []

    def update(frame):
        nonlocal generation
        # Expandieren, bis alle Lebensmittel gefunden sind
        agent.expand_until_food_found()
        
        # Wenn alle Nahrungsquellen gesammelt sind, starte neue Generation
        if agent.has_collected_all_food():
            scores.append(agent.score)
            generation += 1
            print(f"Generation {generation}: Score = {agent.score}, Visited = {len(agent.visited)}")
            
            # Essen neu platzieren und Agenten zurücksetzen
            place_food(labyrinth, food_count=5)
            agent.reset()

        ax.imshow(agent.display_labyrinth(), cmap="viridis")
        ax.set_title(f"Generation: {generation} | Score: {agent.score}")

    ani = FuncAnimation(fig, update, frames=epochs, repeat=False)
    plt.show()

    # Plot der Scores nach dem Training
    plt.figure()
    plt.plot(scores)
    plt.title("Effizienz der Generationen")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.show()

labyrinth = generate_labyrinth(LABYRINTH_SIZE)
train_agent(labyrinth, epochs=100)
