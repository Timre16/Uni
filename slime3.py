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
    # Essen zufällig platzieren
    for _ in range(5):  # 5 Nahrungsquellen
        x, y = random.randint(0, size[0] - 1), random.randint(0, size[1] - 1)
        labyrinth[x, y] = FOOD
    return labyrinth

# Agent-Umgebung
class SlimeAgent:
    def __init__(self, labyrinth):
        self.labyrinth = labyrinth
        self.start_position = (random.randint(0, labyrinth.shape[0] - 1), 
                               random.randint(0, labyrinth.shape[1] - 1))
        self.positions = [self.start_position]  # Alle Positionen des Agenten
        self.food_collected = 0
        self.frontier = [self.start_position]  # Außengrenze des Schleims
        self.visited = set([self.start_position])  # Set für besuchte Positionen
        self.collect_food_positions()

    def reset_position(self):
        self.positions = [self.start_position]
        self.food_collected = 0
        self.visited = set([self.start_position])
        self.frontier = [self.start_position]
        self.collect_food_positions()

    def collect_food_positions(self):
        # Sammle alle Essenspositionen für die Optimierung
        self.reachable_food = set()
        for x in range(LABYRINTH_SIZE[0]):
            for y in range(LABYRINTH_SIZE[1]):
                if self.labyrinth[x, y] == FOOD:
                    self.reachable_food.add((x, y))

    def expand_one_step(self):
        # Alle Positionen an der Außengrenze expandieren
        new_frontier = []
        for current_pos in self.frontier:
            x, y = current_pos

            # Erweiterungsrichtung (obere, untere, linke, rechte Richtung)
            directions = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]  # Oben, Unten, Links, Rechts
            random.shuffle(directions)  # Zufällige Richtung probieren

            for nx, ny in directions:
                if 0 <= nx < LABYRINTH_SIZE[0] and 0 <= ny < LABYRINTH_SIZE[1]:
                    if (nx, ny) not in self.visited and self.labyrinth[nx, ny] != WALL:
                        # Neue Position hinzufügen
                        new_frontier.append((nx, ny))
                        self.visited.add((nx, ny))  # Als besucht markieren
                        if self.labyrinth[nx, ny] == FOOD:
                            self.food_collected += 1
                            self.labyrinth[nx, ny] = EMPTY  # Essen konsumiert
        # Setze die neue Außengrenze
        self.frontier = new_frontier

    def has_collected_all_food(self):
        # Prüfe, ob alle Essensquellen erreicht wurden
        return self.reachable_food.issubset(self.visited)

    def display_labyrinth(self):
        display_labyrinth = np.copy(self.labyrinth)
        for x, y in self.visited:
            display_labyrinth[x, y] = AGENT
        return display_labyrinth

# Haupt-Trainingsfunktion
def train_agent(epochs=1000):
    labyrinth = generate_labyrinth(LABYRINTH_SIZE)
    agent = SlimeAgent(labyrinth)
    
    fig, ax = plt.subplots()
    ax.set_title("Agent lernt das Labyrinth")

    def update(frame):
        # Führe die Expansion aus, bis alle Nahrungsquellen erreicht sind
        agent.expand_one_step()
        
        # Wenn alle Nahrungsquellen gesammelt sind, stoppe
        if agent.has_collected_all_food():
            ax.set_title(f"Finale Generation | Food Collected: {agent.food_collected} | Size: {len(agent.visited)}")
            ani.event_source.stop()  # Stoppt die Animation

        ax.imshow(agent.display_labyrinth(), cmap="viridis")
        ax.set_title(f"Generation: {frame+1} | Food Collected: {agent.food_collected} | Size: {len(agent.visited)}")

    # Animation
    ani = FuncAnimation(fig, update, frames=epochs, repeat=False)
    plt.show()

train_agent(epochs=1000)
